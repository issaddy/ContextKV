import json

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


def test_fixed_arrivals_keep_history_and_slot_wait_in_response_time(tmp_path):
    import asyncio
    import time
    from bench.study_workloads import WorkloadDispatcher, freeze_workload, read_workload_manifest
    from bench.capacity_workflow import full_queue_methods

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
    assert full_queue_methods(1) == ("B6_native", "S3_value")
    assert set(full_queue_methods(4)) == {"B6_native", "S3_native", "B6_value", "S3_value"}
    assert full_queue_methods(1, ["S3_native"]) == ("S3_native",)

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
