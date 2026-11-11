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
    from bench.capacity_workflow import (FULL_QUEUE_METHODS, OPTIONAL_FULL_QUEUE_METHODS,
                                        full_queue_method_specs, full_queue_gap_limits)
    from bench.study import merge_config
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
    expected = (*FULL_QUEUE_METHODS, *OPTIONAL_FULL_QUEUE_METHODS)
    assert FULL_QUEUE_METHODS == ("RAG_native", "context_cache_demand", "LlamaIndex_native")
    from pathlib import Path
    import yaml
    declaration = yaml.safe_load(Path("configs/study_matrix.yaml").read_text())["final_evaluation"]
    assert declaration["baseline"] == "RAG_native"
    assert tuple(declaration["main_methods"]) == FULL_QUEUE_METHODS
    common = {"output_budgets": [512, 2048], "gap_limits": {"max_units": 20, "max_passages": 10}}
    gain_settings = dict(common, gain_asset="measured-B")
    rule_settings = dict(common, action_policy="rule")
    compilation = {"mode": "contextpilot", "checkout": "pinned-author"}
    specs = {spec["id"]: spec for spec in full_queue_method_specs(
        expected, gain_settings, compilation, rule_settings=rule_settings)}
    assert all(spec["overrides"]["execution"]["prefix_retention"] == {"mode": "native"}
               for method, spec in specs.items() if method not in {
                   "context_cache_demand", "context_no_fact_reuse", "Rule_contextpilot_joint_demand"})
    full = specs["context_cache_demand"]["overrides"]
    assert full["execution"]["prefix_retention"]["mode"] == "demand"
    assert full["execution"]["prefix_retention"]["allocation"] == "next_use"
    for pair in declaration["optional_ablations"].values():
        reference, candidate = (specs[pair[key]]["evaluation_factors"] for key in ("reference", "method"))
        changed = {name for name in reference if name != "role" and reference[name] != candidate[name]}
        assert changed == ({"retention", "retention_allocation"} if pair["factor"] == "retention"
                           else {pair["factor"]})
    rule = specs["Rule_contextpilot_joint_demand"]["overrides"]
    assert rule["reading"] == full["reading"]
    assert {k: v for k, v in rule["execution"].items() if k != "plan_gain"} == {
        k: v for k, v in full["execution"].items() if k != "plan_gain"}
    assert {k: v for k, v in rule["execution"]["plan_gain"].items() if k != "action_policy"} == {
        k: v for k, v in full["execution"]["plan_gain"].items() if k != "gain_asset"}
    assert specs["RAG_native"]["overrides"]["execution"]["plan_gain"] is None
    assert specs["LlamaIndex_native"]["overrides"]["execution"]["dialogue_mode"] == "llamaindex"
    control = specs["context_no_fact_reuse"]["overrides"]
    assert control["execution"] == full["execution"]
    assert control["reading"] == {**full["reading"], "fact_reuse": False}
    fixed = full_queue_method_specs(["RAG_native"], None, None)[0]
    assert fixed["overrides"]["execution"]["plan_gain"] is None
    inherited = merge_config({"reading": {"numeric_record_projection": True}}, fixed["overrides"])
    assert inherited["reading"]["numeric_record_projection"] is False
    demand_policy = dict(mode="demand", allocation="paired_value", pricing_asset=str(tmp_path/"pricing.json"),
        ttl_seconds=30, horizon_seconds_choices=[30, 90], protection_budget_tokens=4096,
        lambda_gpu=.25, lambda_host=.1)
    for spec in full_queue_method_specs(
            ["context_cache_demand", "context_no_fact_reuse", "Rule_contextpilot_joint_demand"],
            gain_settings, compilation, rule_settings=rule_settings, demand_policy=demand_policy):
        assert spec["overrides"]["execution"]["prefix_retention"] == demand_policy
        assert spec["evaluation_factors"]["retention_allocation"] == "paired_value"
    next_use = full_queue_method_specs(["context_cache_demand"], gain_settings, compilation,
        demand_policy={**demand_policy, "allocation": "next_use"})[0]
    assert "pricing_asset" not in next_use["overrides"]["execution"]["prefix_retention"]
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


def test_dataset_queue_preparation_uses_selected_split_protocol_and_workload(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    import yaml
    from bench import capacity_workflow as workflow
    from bench.study import dataset_rows
    from bench.study_workloads import read_workload_manifest

    inputs = {}
    for split, count in (('dev', 1), ('test', 2)):
        path = tmp_path/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(dict(task_id=f'{split}:{turn}', conversation_id=split,
            turn=turn, Collection='track', contexts=[], input=[{'speaker': 'user', 'text': 'Question?'}]))+'\n'
            for turn in range(1, count+1)))
        inputs[split] = str(path)
    shared_path, split_path = tmp_path/'shared.jsonl', tmp_path/'splits.json'
    shared_path.write_text(''.join(Path(path).read_text() for path in inputs.values()))
    split_path.write_text(json.dumps({'dev': ['dev'], 'test': ['test']}))
    shared_spec = {'input': str(shared_path), 'split_manifest': str(split_path)}
    assert [row['task_id'] for row in dataset_rows(shared_spec, 'test', task_ids=['test:2'])] == ['test:2']
    with pytest.raises(KeyError, match='dev:1'):
        dataset_rows(shared_spec, 'test', task_ids=['dev:1'])
    run_path, retriever_path = tmp_path/'reader.yaml', tmp_path/'retriever.yaml'
    run_path.write_text('engine: {}\n')
    retriever_path.write_text('{}\n')
    config_path, matrix_path = tmp_path/'study.yaml', tmp_path/'matrix.yaml'
    config_path.write_text(yaml.safe_dump(dict(generation={'run_config': str(run_path)},
        execution={'lambda_time': 0}, reading={'output_tokens_per_fact': 128},
        retrieval={'config_path': str(retriever_path)})))
    spec = dict(inputs=inputs, split_kind='native', primary_metric='RB_llm', normalization=1)
    matrix = dict(seed=42, protocol='P1', selection={'bootstrap_samples': 20},
        resources={'gpu_capacity_tokens': 32, 'host_capacity_tokens': 64},
        final_evaluation={'baseline': 'RAG_native', 'main_methods': list(workflow.FULL_QUEUE_METHODS),
            'acceptance': {'candidate': 'context_cache_demand', 'quality_improvement': .05,
                           'quality_noninferiority': .002, 'p95_ratio': .85}},
        datasets={'track': spec, 'second': {**spec, 'protocol': 'P3', 'inputs': {'test': inputs['dev']}}})
    matrix_path.write_text(yaml.safe_dump(matrix))
    monkeypatch.setattr('controller.config.load_config', lambda _: SimpleNamespace(
        engine=SimpleNamespace(base_url='http://fixture:30000', gpu_capacity_tokens=32, host_capacity_tokens=64),
        model=SimpleNamespace(name='fixed-Qwen', revision='fixed')))
    args = SimpleNamespace(config=str(config_path), matrix=str(matrix_path), split='test', datasets=None,
        cohorts=None, arrival_plan=None, reader_base_url='http://fixture:30000', methods=['RAG_native'],
        gain_asset=None, reader_profile_id='fixed', online_process_roots=(123,), prepare_only=True,
        online_gpu_devices=(0,), lambda_time=None, gap_max_units=None, gap_max_passages=None,
        read_outputs=None, sampling_interval=.5, contextpilot_checkout=None,
        external_reranker='released-reranker', concurrency=[1, 4], repetitions=1,
        complete_slo_seconds=None, project_root='.', replicate_id=None)
    root = tmp_path/'test-run'
    root.mkdir()
    plan = workflow.prepare(args, root)
    assert plan['split'] == 'test' and plan['source_arrival_plan'] is None
    assert len(plan['jobs']) == 4 and plan['requested_qa'] == 6
    assert {(job['dataset'], job['requested_qa'], job['protocol']) for job in plan['jobs']} == {
        ('track', 2, 'P1'), ('second', 1, 'P3')}
    job = next(job for job in plan['jobs'] if job['dataset'] == 'track')
    workload = read_workload_manifest(job['workload'])
    assert workload.dispatch_order == ('test:1', 'test:2')
    assert workload.turns[1].release_after_task_id == 'test:1'
    assert workload.turns[1].think_time_seconds == 0
    assert workload.dispatch_semantics != 'fixed_arrivals'
    assert yaml.safe_load(Path(job['matrix']).read_text())['protocol'] == 'P1'

    # The same runner summarizes the selected test denominator, not the dev cohort.
    directory = Path(job['output'])/job['method']/job['dataset']
    directory.mkdir(parents=True)
    records = [dict(task_id=f'test:{turn}', conversation_id='test', status='completed',
        metrics={'RB_llm': score}, cost={'complete_ms': 100*turn}, request_window=[0, turn*.1],
        response={'output_tokens': 1, 'truncated': False})
        for turn, score in ((1, .5), (2, 1.))]
    (directory/'results.unscored.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    monkeypatch.setattr(workflow, 'paid_execution_index', lambda *args, **kwargs: {})
    result = workflow.summarize_full_queue(job)
    assert result['dataset'] == 'track' and result['split'] == 'test' and result['protocol'] == 'P1'
    assert result['dataset_protocols']['track']['history_protocol'] == 'P1'
    assert result['requested_qa'] == result['completed_qa'] == 2
    assert result['queue_quality'] == .75 and result['p95_complete_ms'] == 195
    assert 'track_weights' not in result

    arrival_path = tmp_path/'arrivals.json'
    arrival_path.write_text(json.dumps({'format': 'contextkv_fixed_arrivals_v1',
        'datasets': {'track': {'name': 'fixed-dev', 'seconds': {'dev:1': 3}}}}))
    args.split, args.datasets, args.arrival_plan = 'dev', ['track'], str(arrival_path)
    fixed_root = tmp_path/'fixed-dev-run'
    fixed_root.mkdir()
    fixed = workflow.prepare(args, fixed_root)
    assert fixed['split'] == 'dev' and {job['requested_qa'] for job in fixed['jobs']} == {1}
    restored = read_workload_manifest(fixed['jobs'][0]['workload'])
    assert restored.dispatch_semantics == 'fixed_arrivals'
    assert restored.dispatch_order == ('dev:1',) and restored.turns[0].start_delay_seconds == 3
