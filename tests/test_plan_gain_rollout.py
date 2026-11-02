"""One CPU protocol smoke; fixture scores are not experimental model results."""
from dataclasses import replace
import json
import time

import httpx
import pytest
import yaml

from bench.mock_engine import MockEngine
from controller.context_schema import (ContextBudget, ContextPlan, EvidenceSet, EvidenceUnit,
                                       Need, SourceSpan, TaskState, TextRef)
from controller.evidence import make_evidence
from controller.schema import OnlineTurn
from controller.serialization import to_jsonable
from controller.service import Controller, EvidenceSelection
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from training.plan_gain_protocol import AuditedTaskInterpreter
from training.plan_gain_rollout import ResourceWindows, collect


class ProtocolReader(MockEngine):
    """Explicit native-wire test double; no GPU/quality measurement claim."""
    def __init__(self):
        super().__init__()
        self._sampling_params = {"temperature": 0}
        self.reads = 0
    async def info(self):
        return replace(await super().info(), engine_kind="sglang")
    async def generate(self, *, response_schema=None, **kwargs):
        if response_schema:
            self.reads += 1
            self.last_read_budget = kwargs["max_new_tokens"]
            props = response_schema["properties"]["claims"]["items"]["properties"]
            self.response_text = json.dumps({"format_version": 2, "claims": [{
                "need_id": "n", "text": "Use the form", "entity": "applicant", "relation": "procedure",
                "value": "form", "unit": "", "conditions": [], "followups": [],
                "span_ids": props["span_ids"]["items"]["enum"][:1], "parent_claim_ids": []}],
                "support": ["n"], "conflicts": [], "open_needs": []})
            if self.reads == 1:
                self.response_text = self.response_text+" "*2048  # completed READ length fault
        else:
            self.response_text = "Direct answer." if not self.reads else "Complete supported answer."
            if self.reads and self.last_read_budget == 1024:
                self.response_text *= 100  # completed ANSWER length fault
        return await super().generate(**kwargs)


class Sampler:
    gpu_devices, process_ids = (0,), (123,)
    def __init__(self):
        self.samples = []
    def sample(self, *args):
        return {"timestamp": time.monotonic(), "gpu_gib": 1, "host_gib": 2}
    async def start(self):
        self.samples.append(self.sample())
    async def stop(self):
        self.samples.append(self.sample())
    def observation(self):
        return {"host_basis": "specified protocol-test PSS", "samples": self.samples}


class Entry:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_one_frozen_state_generates_direct_read_grid_scores_and_importable_actual_delta_packets(tmp_path):
    template = ByteTemplate()
    text = "\n".join("Step "+str(index)+": use the original form." for index in range(1, 9))
    block = make_evidence(text, "source", corpus_id="fixture", source_uri="fixture://source")
    unit = EvidenceUnit("u", (SourceSpan("span", block.key, 0, len(text)),), text)
    state = TaskState("task", "How do I apply?", needs=(Need("n", operation="procedure",
        anchor_spans=(TextRef(1, "user", 0, 3),)),), current_turn_id=1)
    base = ContextPlan("source-plan", state, EvidenceSet("selected", (unit,), source_blocks=(block,)),
        budget=ContextBudget(output_tokens=128, body_tokens=4096, max_steps=64))
    row = {"task_id": "task", "conversation_id": "fixture", "turn": 1, "split": "train",
        "question": state.raw_question, "actual_history": [], "snapshot_id": "source-snapshot",
        "rollout_policy": "dense5_rebuild_qwen3_32b_nonthinking", "reference_answer": "OFFLINE_ONLY_TARGET",
        "script_history_review_status": "reviewed", "script_history_mismatch": False,
        "task_state": to_jsonable(state), "verified_grounding": [
            {"passage_id": "source", "start": 0, "end": 4, "quote": "Step"}],
        "response": {"receipt": {"generation_complete": True, "input_hash": "fixture-source-receipt"}},
        "answer": "Source rollout actual answer."}
    sources = [{"source_id": "source", "text": text, "version": "v1", "source_uri": "fixture://source"}]
    paths = {}
    for name, rows in (("task_labels", [row]), ("actual_rollouts", [row]), ("sources", sources)):
        path = tmp_path/(name+".jsonl")
        path.write_text("".join(json.dumps(item)+"\n" for item in rows))
        paths[name] = str(path)
    config_path = tmp_path/"reader.yaml"
    config_path.write_text(yaml.safe_dump({"engine": {"kind": "mock"},
        "planner": {"strategy": "always_rebuild"}}))
    interpreter = AuditedTaskInterpreter(None)
    class Provider:
        manifest = {"source": "specified-protocol-fixture"}
        support_backend = None
        traces = {}
        def _usage_sources(self):
            return ()
        def _auxiliary_usage(self, *args):
            return {"input_tokens": 0, "output_tokens": 0, "compute_ms": 0.0, "models": []}
        async def retrieve(self, snapshot, query, request):
            from controller.context_schema import RetrievalState
            bound = interpreter.interpret(OnlineTurn(snapshot.session_id, request, 1,
                "fixture", query, (), (), "P3"), RetrievalState(0, snapshot.raw_history))
            assert bound == state
            self.traces[snapshot.session_id, request] = {"pool": "specified-original-fixture"}
            return EvidenceSelection((block,), (block.key,), context_plans=(base,))
    def judge(request):
        if request.url.path == "/context/engine/info":
            return httpx.Response(200, json={"fingerprint": to_jsonable(template.fingerprint)})
        payload = json.loads(request.content)
        prompt = bytes(payload["input_ids"]).decode()
        quality = .8 if '"answer": "Complete supported answer."' in prompt else .2
        return httpx.Response(200, json={"text": json.dumps({
            **{name: quality for name in ("necessary_points", "grounding", "conditions_versions", "task_score")},
            "disposition": "answer", "rationale": "specified protocol fixture"}),
            "meta_info": {"completion_tokens": 24, "finish_reason": {"type": "stop"}}})
    service = Controller(SessionStore(tmp_path/"state.sqlite3"), ProtocolReader(), template=template)
    settings = {**paths, "run_config": str(config_path), "judge_config": str(config_path),
        "judge_base_url": "http://fixture:30000", "reader_profile_id": "specified-protocol-profile",
        "output_dir": str(tmp_path/"collection"), "split": "train", "limit_tasks": 1,
        "partition_seed": "plan-gain-native-train-v1", "dev_fraction": .2, "read_outputs": (512, 1024, 2048),
        "max_generations": 64, "online_gpu_devices": (0,), "judge_gpu_devices": (0,),
        "extra_process_roots": (), "scope_scan": True, "cache_start": "observed"}
    config = {"study": {"strategy": "B7"}, "generation": {"teacher_url": "http://fixture:30000"},
        "execution": {"window_tokens": 32768, "output_tokens": 128, "lambda_time": 0,
                      "lambda_gpu": 0, "lambda_host": 0}, "retrieval": {"corpus_id": "fixture"}}
    async with httpx.AsyncClient(base_url="http://fixture:30000", transport=httpx.MockTransport(judge)) as client:
        manifest = await collect(config, settings, controller=service, provider=Provider(), interpreter=interpreter,
            resources=ResourceWindows(Sampler(), Sampler()), judge_client=client, judge_template=template,
            accounting=Entry())
    output = tmp_path/"collection"
    assert manifest["actual_runs"] == 4 and manifest["actual_gain_pairs"] == 2
    assert manifest["run_statuses"] == {"completed": 3, "failed": 1}
    gains = [json.loads(line) for line in (output/"gains.jsonl").read_text().splitlines()]
    assert [row["delta_q"] for row in gains] == pytest.approx([0, .6])
    registrations = [json.loads(path.read_text()) for path in sorted(output.glob("tasks/*/branch-*/before.json"))]
    runs = [json.loads(path.read_text()) for path in sorted(output.glob("tasks/*/branch-*/run.json"))]
    assert all(registration["registered_at"] < run["actual_branch_execution_cost"]["window"]["start"]
               for registration, run in zip(registrations, runs, strict=True))
    assert len({item["frozen_state_key"] for item in registrations}) == 1
    assert [run["qa_increment"] for run in runs] == [1, 0, 1, 1]
    assert runs[2]["response"]["truncated"] is True
    assert runs[0]["response"]["read_count"] == 0
    assert sum(step["task"]["kind"] == "READ" for step in registrations[1]["plan_candidate"]["steps"]) > sum(
        step["task"]["kind"] == "READ" for step in registrations[3]["plan_candidate"]["steps"])
    assert all(runs[index]["cost"]["total_output_tokens"] > 0 for index in (1, 2))
    assert all("OFFLINE_ONLY_TARGET" not in bytes(call["request"]["input_ids"]).decode()
               for call in service.engine.calls)
    assert all(row["cost"]["gpu_gib_seconds"] > 0 for row in runs)
    from training.plan_gain_completion import finalize
    finalized = tmp_path/"completion-labels"
    completion = finalize(output, finalized)
    assert completion["completion_counts"] == {"1": 3, "0": 1}
    effective = [json.loads(line) for line in (finalized/"effective-gains.jsonl").read_text().splitlines()]
    assert [item["effective_delta_q"] for item in effective] == pytest.approx([-.2, 0, .6])
    assert effective[0]["raw_Q_counterfactual"] is None
    assert effective[0]["utility_counterfactual"] == 0
    await service.aclose()
    service.store.close()
