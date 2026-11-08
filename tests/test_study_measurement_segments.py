import json

import pytest

from bench.study import save_measurement_segment


def test_resumed_measurements_preserve_each_raw_series_and_idle_batch_cost(tmp_path):
    first = save_measurement_segment(tmp_path, "segment-one", physical={"samples": [1, 2]},
        native={"seq": [10, 11]}, batch={"gpu_gib_seconds": 20, "idle_seconds": 5})
    before = {path.name: path.read_bytes() for path in first.iterdir()}
    second = save_measurement_segment(tmp_path, "segment-two", physical={"samples": [3, 4]},
        native={"seq": [12, 13]}, batch={"gpu_gib_seconds": 30, "idle_seconds": 7})
    assert {path.name: path.read_bytes() for path in first.iterdir()} == before
    index = json.loads((tmp_path/"measurement_segments.json").read_text())
    assert set(index["segments"]) == {str(first.relative_to(tmp_path)), str(second.relative_to(tmp_path))}
    assert sum(json.loads((tmp_path/path/"batch_cost.json").read_text())["idle_seconds"]
               for path in index["segments"]) == 12


@pytest.mark.asyncio
async def test_explicit_request_namespace_resumes_failed_p3_without_changing_task_ids(tmp_path, monkeypatch):
    import time
    import yaml
    from adapters import study_retrieval
    from bench import native_metrics, study_resources
    from bench.capacity_workflow import declared_execution_rows
    from bench.mock_engine import MockEngine
    from bench.study import run_dataset, saved_study_rows
    from controller.context_schema import TaskState
    from controller.evidence import make_evidence
    from controller.planner import PlannerKnobs
    from controller.schema import QueryContext
    from controller.service import Controller, EvidenceSelection
    from controller.session_store import SessionStore
    from controller.templates import ByteTemplate

    block = make_evidence("The source supplies the answer.", "source")
    engine, calls = MockEngine(), []

    class Provider:
        corpus_snapshot_id, manifest = "fixture", {"source": "CPU protocol fixture"}

        def __init__(self):
            self.traces = {}

        async def retrieve(self, snapshot, query, task_id):
            calls.append((task_id, len(snapshot.raw_history)))
            if task_id == "task:2" and calls.count((task_id, 1)) == 1:
                raise ValueError("explicit failed input fixture")
            task = TaskState(task_id, query, current_turn_id=len(snapshot.raw_history)+1)
            self.traces[snapshot.session_id, task_id] = {"task_state": {"task_id": task_id}}
            return EvidenceSelection((block,), (block.key,), QueryContext(query, "current", task_state=task))

        def close(self):
            pass

        def finish_request(self, session_id, task_id):
            pass

        def take_trace(self, session_id, task_id):
            return self.traces.pop((session_id, task_id), {})

    class Sampler:
        def __init__(self, *args, **kwargs):
            self.samples = []

        async def start(self):
            self.samples.append({"timestamp": time.monotonic(), "gpu_gib": 1., "host_gib": 1.})

        async def stop(self):
            await self.start()

        def observation(self):
            return {"samples": self.samples}

    def controller(config):
        return Controller(SessionStore(tmp_path/"state/sessions.sqlite3"), engine,
                          template=ByteTemplate(), knobs=PlannerKnobs(strategy="always_rebuild"))

    monkeypatch.setattr("bench.study.create_controller", controller)
    monkeypatch.setattr(study_retrieval, "create_study_retriever", lambda *args: Provider())
    monkeypatch.setattr(study_resources, "StudyResourceSampler", Sampler)
    monkeypatch.setattr(study_resources, "listening_processes", lambda ports: ())
    monkeypatch.setattr(native_metrics, "NativePoolSampler", Sampler)
    monkeypatch.setattr(native_metrics, "pool_series", lambda *args: ((), {}))
    run = tmp_path/"run.yaml"
    run.write_text(yaml.safe_dump({"engine": {"kind": "mock"}}))
    tasks = tmp_path/"tasks.jsonl"
    tasks.write_text("".join(json.dumps(dict(task_id=f"task:{i}", conversation_id="session", turn=i,
        Collection="fixture", contexts=[], input=[{"speaker": "user", "text": f"Question {i}?"}]))+"\n"
        for i in (1, 2, 3)))
    config = {"generation": {"run_config": str(run)}, "task_model": {"mode": "fixture"},
        "support_model": {"mode": "fixture"}, "study": {"strategy": "B0"},
        "execution": dict(window_tokens=32768, output_tokens=1024, lambda_time=0, lambda_gpu=0, lambda_host=0)}
    matrix = {"datasets": {"track": {"input": str(tasks), "continue_conversation_failures": True}},
              "resources": {"gpu_devices": [], "process_ids": []}}
    method = {"id": "B0", "strategy": "B0"}
    initial = await run_dataset(config, matrix, method, "track", "train", tmp_path)
    assert [row["task_id"] for row in initial] == ["task:1"]
    first_segments = json.loads((tmp_path/"measurement_segments.json").read_text())["segments"]
    first_raw = {p: (tmp_path/p/"generation_manifest.json").read_bytes() for p in first_segments}
    failed = tmp_path/"failed_conversations.json"
    original_failure = failed.read_bytes()
    await run_dataset(config, matrix, method, "track", "train", tmp_path)
    assert failed.read_bytes() == original_failure
    indexed, _, _ = declared_execution_rows({"output": tmp_path.parent, "method": tmp_path.name,
                                               "datasets": ["."]})
    assert {row["task_id"]: row["status"] for row in indexed} == {
        "task:1": "completed", "task:2": "failed", "task:3": "blocked"}
    rows = await run_dataset(config, matrix, method, "track", "train", tmp_path,
                             limit_turns=2, request_namespace="condition-schema-v1")
    assert calls == [("task:1", 0), ("task:2", 1), ("task:2", 1)]
    assert [row["task_id"] for row in rows] == ["task:1", "task:2"]
    assert rows[1]["request_id"] == "condition-schema-v1:task:2"
    assert rows[1]["task_state"]["task_id"] == "task:2"
    assert [json.loads(line)["task_id"] for line in (tmp_path/"official.jsonl").read_text().splitlines()] == ["task:1", "task:2"]
    assert all((tmp_path/p/"generation_manifest.json").read_bytes() == raw for p, raw in first_raw.items())
    store = SessionStore(tmp_path/"state/sessions.sqlite3")
    session = "study:B0:track:train:session"
    assert store.find_request(session, "task:2").stage == "FAILED"
    assert store.find_request(session, "condition-schema-v1:task:2").stage == "COMMITTED"
    assert len(store.get_snapshot(session).raw_history) == 2
    manifest = json.loads((tmp_path/"generation_manifest.json").read_text())
    assert manifest["request_namespace"] == "condition-schema-v1"
    assert manifest["task_ids"] == ["task:1", "task:2", "task:3"]
    assert manifest["scheduled_task_ids"] == ["task:1", "task:2"]
    export = tmp_path/"results.unscored.jsonl"
    export.write_text(export.read_text().splitlines()[0]+"\n")
    assert [row["task_id"] for row in saved_study_rows(tmp_path)] == ["task:1", "task:2"]
    store.close()


def test_fixed_arrivals_keep_history_and_slot_wait_in_response_time(tmp_path):
    import asyncio
    import time
    from bench.study_workloads import WorkloadDispatcher, freeze_workload, read_workload_manifest
    from bench.capacity_workflow import full_queue_methods, full_queue_method_specs, full_queue_gap_limits
    from types import SimpleNamespace

    rows = [dict(task_id=f"session:{turn}", conversation_id="session", turn=turn,
                 Collection="track", contexts=[]) for turn in (1, 2)]
    manifest = freeze_workload(rows, name="fixed", concurrency=4,
        fixed_arrivals={"session:1": 0, "session:2": 1})
    path = tmp_path/"workload.json"
    manifest.write(path)
    index = tmp_path/"index.json"
    index.write_text(json.dumps({"format": "contextkv_dataset_workload_index_v1",
                                 "datasets": {"track": str(path)}}))
    assert read_workload_manifest(index)["track"] == manifest
    expected = ("B6_native", "S3_native", "B6_contextpilot", "S3_contextpilot",
                "S3_contextpilot_joint_native", "S3_contextpilot_joint_demand",
                "LlamaIndex_native", "S3_contextpilot_no_fact_reuse")
    assert full_queue_methods(1) == full_queue_methods(4) == expected
    from pathlib import Path
    import yaml
    declaration = yaml.safe_load(Path("configs/study_matrix.yaml").read_text())["final_evaluation"]
    assert tuple(cell["method"] for cell in declaration["factorial"].values()) == expected[:4]
    assert tuple(cell["method"] for cell in declaration["joint_context"].values()) == expected[4:6]
    specs = {spec["id"]: spec for spec in full_queue_method_specs(expected,
        {"gain_asset": "measured-B"}, {"mode": "contextpilot", "checkout": "pinned-author"})}
    assert all(spec["overrides"]["execution"]["prefix_retention"] == {"mode": "native"}
               for method, spec in specs.items() if method != expected[5])
    assert specs[expected[5]]["overrides"]["execution"]["prefix_retention"]["mode"] == "demand"
    assert specs["B6_native"]["overrides"]["execution"]["plan_gain"] is None
    assert specs["B6_contextpilot"]["overrides"]["execution"]["cache_compilation"]["mode"] == "contextpilot"
    assert specs["LlamaIndex_native"]["overrides"]["execution"]["dialogue_mode"] == "llamaindex"
    control = specs["S3_contextpilot_no_fact_reuse"]["overrides"]
    assert control["execution"] == specs["S3_contextpilot"]["overrides"]["execution"]
    assert control["reading"] == {"fact_reuse": False}
    fixed = full_queue_method_specs(["B6_native"], None, None)[0]
    assert fixed["overrides"]["execution"]["plan_gain"] is None
    assert full_queue_methods(1, ["S3_native"]) == ("S3_native",)
    gain = SimpleNamespace(provenance={"continuation_gap_limits": {"max_units": 12, "max_passages": 12}})
    execution = {"state_observation": {"gap_limits": {"max_units": 30, "max_passages": 8}}}
    assert full_queue_gap_limits({"state_observation": None}, gain) == gain.provenance["continuation_gap_limits"]
    assert full_queue_gap_limits(execution, gain, max_units=20) == {"max_units": 20, "max_passages": 12}
    assert full_queue_gap_limits(execution, gain, max_passages=9) == {"max_units": 12, "max_passages": 9}
    assert full_queue_gap_limits(execution) == {"max_units": 30, "max_passages": 8}
    assert full_queue_gap_limits({"state_observation": None}) == {"max_units": 96, "max_passages": 10}

    async def run():
        dispatcher = WorkloadDispatcher(manifest)
        dispatcher.origin = time.monotonic()-100
        await dispatcher.wait_turn(rows[0], ())
        pending = asyncio.create_task(dispatcher.wait_turn(rows[1], ("own committed QA",)))
        await asyncio.sleep(0)
        assert not pending.done()
        dispatcher.complete_turn("session:1")
        second = await pending
        assert second["actual_arrival_seconds"] == second["planned_arrival_seconds"] == 1
        assert second["history_ready_seconds"] >= 100
        # The runner uses this original timestamp before waiting for a slot.
        start = dispatcher.origin+second["actual_arrival_seconds"]
        semaphore = asyncio.Semaphore(0)
        async def submit():
            async with semaphore:
                return time.monotonic()-start
        queued = asyncio.create_task(submit())
        await asyncio.sleep(0)
        assert not queued.done()
        semaphore.release()
        assert await queued >= 99
    asyncio.run(run())


def test_cancelled_fixed_tail_remains_observed_and_counts_blocked_requests():
    import asyncio
    import time
    from bench.capacity_workflow import delivery_summary
    from bench.study_workloads import WorkloadDispatcher, freeze_workload

    async def run():
        rows = [dict(task_id=f"client:{turn}", conversation_id="client", turn=turn,
                     Collection="track", contexts=[]) for turn in (1, 2, 3)]
        manifest = freeze_workload(rows, name="cancelled-tail", concurrency=4,
            fixed_arrivals={"client:1": 0, "client:2": .01, "client:3": .03})
        dispatcher = WorkloadDispatcher(manifest)
        await dispatcher.wait_turn(rows[0], ())
        failed_end = time.monotonic()
        await dispatcher.cancel_conversation("client", "first request failed")
        ticks = []
        observing = True
        async def sampler():
            while observing:
                ticks.append(time.monotonic())
                await asyncio.sleep(.001)
        sampling = asyncio.create_task(sampler())
        window = await dispatcher.finish_arrivals()
        observing = False
        await sampling
        assert window["end"] >= dispatcher.origin+.03
        assert failed_end < dispatcher.origin+.03
        assert ticks[-1] > failed_end
        metrics = delivery_summary([], [dict(conversation_id="client", task_id="client:1",
            request_window=[dispatcher.origin, failed_end])],
            dict(origin=dispatcher.origin, cancelled=dispatcher.cancelled, observation_window=window),
            complete_slo_seconds=1)
        assert metrics["observation_seconds"] >= .03
        assert metrics["statuses"] == {"blocked": 2, "failed": 1}
        assert metrics["failure_fraction"] == 1
        assert metrics["latency_slo_goodput_requests_per_second"] == 0
    asyncio.run(run())
