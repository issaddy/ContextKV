from dataclasses import replace
import json
from pathlib import Path
import time

import pytest
import yaml

from bench import reader_diagnosis as diagnosis
from bench.evidence_diagnosis import OracleAnnotations, review_binding
from bench.mock_engine import MockEngine
from controller.context_schema import EvidenceSet, Need, ReadTask, VersionIntent
from controller.reading import build_read_plan
from controller.schema import SessionSnapshot
from controller.serialization import to_jsonable
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from test_evidence_diagnosis import audited_case, original_unit


class ServedFixtureEngine(MockEngine):
    async def generate(self, *, response_schema=None, **kwargs):
        if response_schema:
            need = response_schema["properties"]["support"]["items"].get("enum", ["need"])[0]
            spans = response_schema["properties"]["claims"]["items"]["properties"]["span_ids"]["items"]["enum"]
            self.response_text = json.dumps({"format_version": 2, "claims": [{"need_id": need, "text": "Use the current Linux form",
                "entity": "applicant", "relation": "procedure", "value": "form L", "unit": "",
                "conditions": [], "followups": [], "span_ids": spans, "parent_claim_ids": []}],
                "support": [need], "conflicts": [], "open_needs": []})
        else:
            self.response_text = "The current Linux applicant submits form L."
        return await super().generate(**kwargs)


class PhysicalFixture:
    def __init__(self, *args):
        self.samples = []
    async def start(self):
        self.samples.append({"timestamp": time.monotonic(), "gpu_gib": 1.0, "host_gib": 2.0})
    async def stop(self):
        self.samples.append({"timestamp": time.monotonic(), "gpu_gib": 1.0, "host_gib": 2.0})
    def observation(self):
        return {"source": "explicit_unit_test_fixture", "samples": self.samples}


class NativeFixture:
    def __init__(self, *args, **kwargs):
        self.samples = []
    async def start(self):
        self.samples.append({"timestamp": time.monotonic(), "gpu_slots": 0, "host_slots": 0})
    async def stop(self):
        self.samples.append({"timestamp": time.monotonic(), "gpu_slots": 10, "host_slots": 0})
    def observation(self):
        return {"source": "explicit_unit_test_fixture", "samples": self.samples, "measurement": "fixture_slots",
                "interval_seconds": .5, "errors": [], "scope": "unit_test"}


def packet(*, oversized_e0=False, verified=False, reading_enabled=True):
    e0, pool, annotations, history, full, scope = audited_case()
    e0 = replace(e0, task_state=replace(e0.task_state, needs=(Need("need"),)),
        selected_set=replace(e0.selected_set, selected_units=(pool.units[0], pool.units[2])))
    annotations = OracleAnnotations(**review_binding(e0, history), requirements=annotations.requirements,
                                    review_origin=annotations.review_origin)
    if oversized_e0:
        block = full.source_blocks[0]
        whole = original_unit("original-8k-selection", block, 0, len(block.text))
        e0 = replace(e0, selected_set=EvidenceSet("frozen-big-E0", (whole,), source_blocks=(block,)))
    if reading_enabled:
        e0 = build_read_plan(e0.task_state, e0.selected_set, e0.coverage_scope, e0.budget,
            template=ByteTemplate(), snapshot=SessionSnapshot("frozen", raw_history=history))
    value = {"task_id": "frozen-task", "actual_history": to_jsonable(history),
        "e0_plan": to_jsonable(e0), "candidate_pool": to_jsonable(pool),
        "full_applicable_pool": to_jsonable(full), "annotations": to_jsonable(annotations),
        "scope_review": to_jsonable(scope), "split": "dev"}
    if verified:
        value["verified_task_state"] = to_jsonable(replace(e0.task_state,
            version_intent=VersionIntent("explicit", version_ids=("2024",))))
    return value


def setup_runtime(tmp_path, monkeypatch, *, engine_class=ServedFixtureEngine):
    run_path = tmp_path/"run.yaml"
    run_path.write_text(yaml.safe_dump({"engine": {"kind": "mock", "base_url": "http://runtime.fixture:30000"}}))
    services = []
    def factory(run):
        template = ByteTemplate()
        service = Controller(SessionStore(Path(run.state_dir)/"sessions.sqlite3"),
            engine_class(fingerprint=template.fingerprint), template=template, policy=run.context,
            knobs=run.planner, prefix_settings=run.prefix_retention)
        services.append(service)
        return service
    monkeypatch.setattr(diagnosis, "create_controller", factory)
    monkeypatch.setattr(diagnosis, "StudyResourceSampler", PhysicalFixture)
    monkeypatch.setattr(diagnosis, "NativePoolSampler", NativeFixture)
    monkeypatch.setattr(diagnosis, "listening_processes", lambda ports: ())
    return str(run_path), services


def test_diagnosis_preserves_original_read_packing_and_layout():
    source = packet()
    source["e0_plan"]["read_layout"] = "source_first"
    source["e0_plan"]["read_packing"] = "source_windows"
    protocol = diagnosis.DiagnosisProtocol(100000)
    template = ByteTemplate()
    prepared = diagnosis.prepare_packet(source, template.tokenizer, protocol)
    plan = diagnosis.build_cell_plan(prepared, "E1", "inferred", protocol, template, SessionSnapshot("s"))
    assert plan.read_layout == "source_first" and plan.read_packing == "source_windows"


@pytest.mark.asyncio
async def test_b3_direct_answer_six_cells_never_add_read_or_reduce_depth(tmp_path, monkeypatch):
    config, _ = setup_runtime(tmp_path, monkeypatch)
    source = packet(reading_enabled=False)
    source["e0_plan"]["layouts"] = ["need_order", "stable_evidence"]
    rows = await diagnosis.run_diagnosis([source], (
        diagnosis.ReaderSpec("R0", config, (0,)), diagnosis.ReaderSpec("R1", config, (1,))),
        tmp_path/"b3-direct", diagnosis.DiagnosisProtocol(100000, cache_start="observed"))
    assert len(rows) == 6 and all(row["status"] == "completed" for row in rows)
    assert all(row["response"]["read_count"] == row["response"]["reduce_count"] == 0 for row in rows)
    assert all(not row["steps"] and not row["context_plan"]["steps"] for row in rows)
    assert all(row["qa_increment"] == 1 and row["reading_enabled"] is False for row in rows)
    e0 = [row for row in rows if row["evidence_level"] == "E0"]
    assert all(row["context_plan"]["layouts"] == source["e0_plan"]["layouts"] for row in e0)
    assert all(row["initial_plan_report"]["direct_answer_inputs_preserved"] for row in e0)


@pytest.mark.asyncio
async def test_old_runtime_claims_are_not_replayed_without_an_initial_plan(tmp_path, monkeypatch):
    config, _ = setup_runtime(tmp_path, monkeypatch)
    source = packet()
    source["e0_plan"]["steps"][0]["input_claim_ids"] = ["old-r0:claim:0"]
    rows = await diagnosis.run_diagnosis([source], (
        diagnosis.ReaderSpec("R0", config, (0,)), diagnosis.ReaderSpec("R1", None, ())),
        tmp_path/"missing-initial", diagnosis.DiagnosisProtocol(100000, cache_start="observed"))
    assert all(row["status"] == "initial_plan_unavailable" for row in rows if row["reader"] == "R0")
    assert all("answer" not in row for row in rows)


@pytest.mark.asyncio
async def test_expanded_gap_plan_is_retained_as_metadata_and_not_executed_as_initial_dag(tmp_path, monkeypatch):
    from copy import deepcopy
    config, _ = setup_runtime(tmp_path, monkeypatch)
    source = packet()
    initial = deepcopy(source["e0_plan"])
    expanded = deepcopy(initial)
    expanded["steps"].append(to_jsonable(ReadTask("gap:1:read:0", "READ", ("need",),
        input_unit_ids=(initial["selected_set"]["selected_units"][0]["unit_id"],),
        input_claim_ids=("read:0:claim:0",), predecessor_step_ids=("read:0",))))
    source.update(e0_plan=expanded, e0_initial_plan=initial, e0_actual_expanded_plan=expanded,
                  e0_plan_stage="initial", e0_plan_origin="specified_initial_trace")
    rows = await diagnosis.run_diagnosis([source], (
        diagnosis.ReaderSpec("R0", config, (0,)), diagnosis.ReaderSpec("R1", config, (1,))),
        tmp_path/"expanded", diagnosis.DiagnosisProtocol(100000, cache_start="observed"))
    assert all(row["response"]["read_count"] == 1 for row in rows)
    assert all("gap:1:read:0" not in {step["step_id"] for step in row["steps"]} for row in rows)
    assert all(row["initial_plan_report"]["actual_expansion_step_ids"] == ["gap:1:read:0"] for row in rows)


@pytest.mark.asyncio
async def test_real_controller_six_cells_share_frozen_history_and_keep_scores_empty(tmp_path, monkeypatch):
    config, services = setup_runtime(tmp_path, monkeypatch)
    source = packet()
    output = tmp_path/"diagnosis"
    rows = await diagnosis.run_diagnosis([source], (
        diagnosis.ReaderSpec("R0", config, (0,), "reader-0"),
        diagnosis.ReaderSpec("R1", config, (1,), "reader-1")), output,
        diagnosis.DiagnosisProtocol(100000, cache_start="observed"))
    assert len(rows) == 6 and all(row["status"] == "completed" for row in rows)
    assert all(row["metrics"] == {} and row["qa_increment"] == row["version_increment"] == 1 for row in rows)
    assert all(row["actual_history"] == source["actual_history"] for row in rows)
    assert all(row["cost"]["gpu_gib_seconds"] > 0 and row["cost"]["complete_window_coverage"] for row in rows)
    assert all(row["steps"] and row["response"]["read_count"] >= 1 for row in rows)
    assert all(row["initial_plan_report"]["initial_dag_preserved"] and
        row["initial_plan_report"]["initial_read_inputs_preserved"] for row in rows if row["evidence_level"] == "E0")
    assert all(row["response"]["action"] == "REBUILD" for row in rows)
    assert all(row["measurement_kind"] == "simulation" for row in rows)
    for row in rows:
        directory = Path(row["generation_dir"])
        assert (directory/"official.jsonl").exists() and (directory/"audit.jsonl").exists()
        assert json.loads((directory/"references.jsonl").read_text())["task_id"] == source["task_id"]
        assert row["response"]["total_input_tokens"] == row["response"]["reader_input_tokens"]
    matrix = json.loads((output/"matrix.json").read_text())
    assert matrix["execution_status"] == "executed_unscored" and matrix["six_grid_passed"] is False
    assert len(services) == 2


@pytest.mark.asyncio
async def test_r1_pending_and_verified_binding_are_separate_matrix_columns(tmp_path, monkeypatch):
    config, _ = setup_runtime(tmp_path, monkeypatch)
    rows = await diagnosis.run_diagnosis([packet(verified=True)], (
        diagnosis.ReaderSpec("R0", config, (0,)), diagnosis.ReaderSpec("R1", None, ())),
        tmp_path/"pending", diagnosis.DiagnosisProtocol(100000, cache_start="observed"))
    assert len(rows) == 12
    assert sum(row["status"] == "completed" for row in rows) == 6
    assert all(row["status"] == "pending" for row in rows if row["reader"] == "R1")
    assert all(row["context_plan"]["task_state"]["version_intent"]["mode"] == "explicit"
        for row in rows if row["reader"] == "R0" and row["binding"] == "verified")
    assert all(row["metrics"] == {} for row in rows)


@pytest.mark.asyncio
async def test_incomplete_summary_review_runs_e0_without_inventing_oracle_cells(tmp_path, monkeypatch):
    config, _ = setup_runtime(tmp_path, monkeypatch)
    source = packet()
    source.update(review_status="review_incomplete", review_reason="generic_summary_has_no_complete_witness")
    source["scope_review"]["complete"] = False
    rows = await diagnosis.run_diagnosis([source], (
        diagnosis.ReaderSpec("R0", config, (0,)), diagnosis.ReaderSpec("R1", None, ())),
        tmp_path/"incomplete", diagnosis.DiagnosisProtocol(100000, cache_start="observed"))
    r0 = {row["evidence_level"]: row for row in rows if row["reader"] == "R0"}
    assert r0["E0"]["status"] == "completed"
    assert r0["E1"]["status"] == r0["E2"]["status"] == "review_incomplete"
    assert "answer" not in r0["E1"] and "answer" not in r0["E2"]


@pytest.mark.asyncio
async def test_oversized_frozen_e0_is_infeasible_instead_of_relabelled_as_smaller_budget(tmp_path, monkeypatch):
    config, _ = setup_runtime(tmp_path, monkeypatch)
    source = packet(oversized_e0=True)
    prepared = diagnosis.prepare_packet(source, ByteTemplate().tokenizer, diagnosis.DiagnosisProtocol(100000))
    budget = prepared["selections"]["E2"].evidence_set.token_cost
    rows = await diagnosis.run_diagnosis([source], (diagnosis.ReaderSpec("R0", config, (0,)),
        diagnosis.ReaderSpec("R1", None, ())), tmp_path/"budget", diagnosis.DiagnosisProtocol(budget, cache_start="observed"))
    e0 = next(row for row in rows if row["reader"] == "R0" and row["evidence_level"] == "E0")
    assert e0["status"] == "infeasible" and e0["body_tokens"] > budget
    assert e0["original_e0_plan"] == source["e0_plan"]
    assert "answer" not in e0


@pytest.mark.asyncio
async def test_failed_real_step_retains_raw_receipt_cost_and_uncompleted_matrix(tmp_path, monkeypatch):
    class MalformedRead(ServedFixtureEngine):
        async def generate(self, *, response_schema=None, **kwargs):
            if response_schema:
                self.response_text = "{bad json"
                return await MockEngine.generate(self, **kwargs)
            return await super().generate(**kwargs)
    config, _ = setup_runtime(tmp_path, monkeypatch, engine_class=MalformedRead)
    output = tmp_path/"failed"
    with pytest.raises(ValueError):
        await diagnosis.run_diagnosis([packet()], (diagnosis.ReaderSpec("R0", config, (0,)),
            diagnosis.ReaderSpec("R1", None, ())), output, diagnosis.DiagnosisProtocol(100000, cache_start="observed"))
    matrix = json.loads((output/"matrix.json").read_text())
    failed = next(row for row in matrix["cells"] if row["status"] == "failed")
    assert failed["qa_increment"] == 0 and failed["steps"][0]["receipt"]["generation_complete"]
    assert failed["steps"][0]["output_ids"]
    assert failed["cost"]["gpu_gib_seconds"] > 0
    assert matrix["execution_status"] == "partial" and matrix["six_grid_passed"] is False
