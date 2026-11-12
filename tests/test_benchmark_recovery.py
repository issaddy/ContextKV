"""Exercise the real dataset runner and arrival dispatcher without a model/GPU."""
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from bench.capacity_workflow import declared_execution_rows, full_queue_method_specs, prepare
from bench.study import close_failed_session, run_dataset
from bench.study_workloads import WorkloadDispatcher, freeze_workload
from controller.config import RunConfig
from controller.context_schema import CompletionReceipt, ReadTask, StepRecord
from controller.engine_client import RetireAck
from controller.prefix_retention import PrefixOwnerRecord
from controller.schema import ConfigFingerprint, ContextEpoch
from controller.schema import TurnRecord
from controller.service import Controller
from controller.session_store import SessionStore
from bench.mock_engine import MockEngine


def source_rows(conversation="conversation", count=4):
    rows = []
    for turn in range(1, count+1):
        history = [message for index in range(1, turn) for message in (
            {"speaker": "user", "text": f"question-{index}"},
            {"speaker": "agent", "text": f"author-{index}"})]
        rows.append({"conversation_id": conversation, "task_id": f"{conversation}:{turn}",
            "turn": turn, "Collection": "collection", "contexts": [],
            "input": [*history, {"speaker": "user", "text": f"question-{turn}"}],
            "targets": [{"text": "hidden-target-must-never-be-passed-online"}]})
    return rows


class FakeStore:
    def __init__(self):
        self.snapshots, self.attempts = {}, {}
        self.closed = set()
        self.syncs = []

    def read_snapshot(self, session, corpus=None):
        return self.snapshots.setdefault(session, SimpleNamespace(raw_history=(), archive=(), version=0))

    def get_snapshot(self, session):
        return self.snapshots[session]

    def sync_history(self, session, history, version):
        if session in self.closed:
            raise RuntimeError("cannot import history into a closed session")
        snapshot = self.snapshots[session]
        assert snapshot.version == version
        snapshot.raw_history, snapshot.version = history, version+1
        self.syncs.append((session, history))
        return snapshot

    def reopen_session(self, session):
        self.closed.discard(session)

    def find_request(self, session, request):
        return self.attempts.get((session, request))

    def get_attempt(self, identity):
        return next(value for value in self.attempts.values() if value.attempt_id == identity)

    def execution_steps_with_stage(self, identity):
        return ()

    def auxiliary_calls(self, identity):
        return ({"call_id": identity+":provider", "kind": "provider_usage", "status": "RECORDED",
                 "input_tokens": 100, "output_tokens": 7, "compute_ms": 5},)

    def reading_progress(self, identity):
        return None

    def outbox(self, session):
        return ()

    def prefix_owners(self, session, **kwargs):
        return ()

    def close(self):
        pass


class FakeEngine:
    async def info(self):
        return {"kind": "simulation"}

    async def metrics(self):
        return {"pool": {"gpu_capacity_tokens": 1000, "host_capacity_tokens": 2000}}


class FakeService:
    def __init__(self, fail_tasks, *, unresolved=False):
        self.store, self.engine = FakeStore(), FakeEngine()
        self.state_observer, self.template = None, None
        self.fail_tasks, self.unresolved = set(fail_tasks), unresolved
        self.calls, self.closed = [], []

    def bind_resident_models(self, manifest):
        pass

    async def close_session(self, session):
        attempts = [value for (sid, _), value in self.store.attempts.items() if sid == session]
        if any(value.stage == "SUBMITTED" for value in attempts):
            raise RuntimeError("native outcome unresolved")
        self.closed.append(session)
        self.store.closed.add(session)

    async def aclose(self):
        pass


class FakeProvider:
    manifest = {"simulation": True}
    corpus_snapshot_id = "corpus"

    def take_trace(self, session, task):
        return {}

    def close(self):
        pass


class FakeQA:
    def __init__(self, service, provider):
        self.service = service

    async def start(self):
        pass

    async def ask(self, session, query, request, *, task_id, **kwargs):
        service, store = self.service, self.service.store
        if session in store.closed:
            raise RuntimeError("closed session")
        snapshot = store.read_snapshot(session)
        service.calls.append((task_id, query, snapshot.raw_history))
        candidate = SimpleNamespace(required_blocks=frozenset(), input_ids=(1, 2), reference_map=())
        attempt = SimpleNamespace(attempt_id="attempt:"+task_id, query=query, stage="STARTED",
            plan=SimpleNamespace(candidate=candidate), query_context=None, reading_outcome=None,
            context_plan=None, result=None, terminal_receipt=None, final_step_id=None,
            failed_from_stage=None, auxiliary_usage=None)
        store.attempts[(session, request)] = attempt
        await asyncio.sleep(0)
        if task_id in service.fail_tasks:
            attempt.stage = "SUBMITTED" if service.unresolved else "FAILED"
            attempt.failed_from_stage = "STARTED"
            raise ValueError("injected model failure")
        answer = "model:"+task_id
        snapshot.raw_history = (*snapshot.raw_history, TurnRecord(len(snapshot.raw_history)+1, query, answer))
        snapshot.version += 1
        attempt.stage = "COMMITTED"
        return SimpleNamespace(attempt_id=attempt.attempt_id, assistant_text=answer,
            visible_evidence_manifest=(), output_ids=(8,), truncated=False, citation_scope=(),
            step_metrics=(), engine_elapsed_ms=10, total_input_tokens=102, total_output_tokens=8,
            reader_input_tokens=2, reader_output_tokens=1, auxiliary_input_tokens=100,
            auxiliary_output_tokens=7, auxiliary_model_ms=5, total_prefill_tokens=2,
            total_device_hit_tokens=0, total_host_loaded_tokens=0, action="REBUILD", output_tokens=1,
            read_count=0, reduce_count=0)


class FakeSampler:
    def __init__(self, *args):
        self.samples = []

    async def start(self):
        self.samples.append({"timestamp": time.monotonic(), "gpu_gib": 1, "host_gib": 1})

    async def stop(self):
        self.samples.append({"timestamp": time.monotonic(), "gpu_gib": 1, "host_gib": 1})


class PendingRetirementEngine(MockEngine):
    retirement_ready = False

    async def retire(self, engine_session_id, expected_generation, operation_id, expected_restart_id, reason):
        if self.retirement_ready:
            return await super().retire(engine_session_id, expected_generation, operation_id, expected_restart_id, reason)
        return RetireAck(engine_session_id, expected_generation, expected_restart_id, operation_id,
                         state="RETIRING", acknowledged=False)


class PaidFailureQA:
    calls = []

    def __init__(self, service, provider):
        self.service = service

    async def start(self):
        pass

    async def ask(self, session, query, request, *, task_id, **kwargs):
        self.calls.append(task_id)
        service, store = self.service, self.service.store
        snapshot = store.read_snapshot(session)
        attempt = store.begin_attempt(session, request, "request-hash", query, snapshot.version)
        info = await service.engine.info()
        epoch_id = "native:"+attempt.attempt_id
        ack = await service.engine.prepare(epoch_id, "prepare:"+epoch_id, info.engine_restart_id)
        epoch = ContextEpoch(session, 0, epoch_id, ack.session_generation, engine_restart_id=info.engine_restart_id)
        generated = await service.engine.generate(input_ids=(1, 2, 3), session_id=epoch_id,
            expected_generation=epoch.session_generation, expected_restart_id=info.engine_restart_id,
            attempt_id=attempt.attempt_id+":read", fingerprint=service.template.fingerprint, max_new_tokens=5)
        store.save_execution_step(StepRecord(attempt.attempt_id, "read", ReadTask("read", "READ", ()), epoch,
            input_ids=(1, 2, 3), output_ids=generated.output_ids, receipt=generated.receipt),
            "RESULT_READY", generated)
        store.fail_attempt(attempt.attempt_id, "injected paid extraction failure")
        raise ValueError("injected paid extraction failure")


class DatasetFailureRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = TemporaryDirectory()
        self.directory = Path(self.temporary.name)/"method"/"dataset"

    async def asyncTearDown(self):
        self.temporary.cleanup()

    async def run_queue(self, rows, protocol, service, *, fixed=False, scheduled=True, qa_class=FakeQA):
        matrix = {"protocol": protocol, "datasets": {"dataset": {"continue_conversation_failures": True}},
            "resources": {"gpu_devices": []}, "cohorts": {"test": {"dataset": [row["task_id"] for row in rows]}}}
        config = {"generation": {"run_config": "unused"}, "task_model": {}, "support_model": {}, "execution": {}}
        workload = (freeze_workload(rows, name="test", concurrency=2,
            fixed_arrivals={row["task_id"]: 0 for row in rows} if fixed else None) if scheduled else None)
        with ExitStack() as stack:
            for target, value in (
                ("bench.study.dataset_rows", lambda *args, **kwargs: rows),
                ("bench.study.load_config", lambda *args: RunConfig()),
                ("bench.study.apply_study_config", lambda run, config: run),
                ("bench.study.create_controller", lambda run: service),
                ("bench.study.export_evaluation", lambda *args, **kwargs: None),
                ("adapters.study_retrieval.create_study_retriever", lambda *args: FakeProvider()),
                ("controller.qa.DocumentQA", qa_class),
                ("bench.study_resources.StudyResourceSampler", FakeSampler),
                ("bench.study_resources.listening_processes", lambda ports: ())):
                stack.enter_context(patch(target, value))
            return await asyncio.wait_for(run_dataset(config, matrix, {"id": "method", "strategy": "B7"},
                "dataset", "test", self.directory, concurrency=2, workload=workload), timeout=3)

    def failures(self):
        return json.loads((self.directory/"failed_conversations.json").read_text())

    def metrics(self):
        return [json.loads(line) for line in (self.directory/"task_metrics.jsonl").read_text().splitlines()]

    async def test_p1_multiple_failures_do_not_block_author_history_turns(self):
        rows = source_rows()
        service = FakeService({"conversation:1", "conversation:3"})
        results = await self.run_queue(rows, "P1", service)
        self.assertEqual([row["task_id"] for row in results], ["conversation:2", "conversation:4"])
        self.assertEqual([task for task, _, _ in service.calls], [row["task_id"] for row in rows])
        self.assertEqual([turn.assistant_text for turn in service.calls[3][2]], ["author-1", "author-2", "author-3"])
        failed = self.failures()
        self.assertEqual(len(failed), 2)
        self.assertTrue(all(not row["blocked_task_ids"] for row in failed))
        self.assertTrue(all(row["cost"]["total_input_tokens"] == 100 for row in failed))
        self.assertTrue(all(row["cost"]["gpu_gib_seconds"] > 0 for row in failed))
        dispatch = json.loads((self.directory/"workload_dispatch.json").read_text())
        self.assertEqual(dispatch["cancelled"], {})
        self.assertEqual(set(dispatch["failed"]), {"conversation:1", "conversation:3"})
        self.assertEqual(set(dispatch["completion_seconds"]), {"conversation:2", "conversation:4"})
        self.assertEqual(len(self.metrics()), 4)
        self.assertEqual(len(service.closed), 3)

    async def test_p3_failure_keeps_real_dependency_and_other_conversation_runs(self):
        rows = [*source_rows("a", 3), *source_rows("b", 2)]
        service = FakeService({"a:1"})
        results = await self.run_queue(rows, "P3", service)
        self.assertEqual({row["task_id"] for row in results}, {"b:1", "b:2"})
        self.assertEqual({task for task, _, _ in service.calls}, {"a:1", "b:1", "b:2"})
        self.assertEqual(self.failures()[0]["blocked_task_ids"], ["a:2", "a:3"])
        b2 = next(history for task, _, history in service.calls if task == "b:2")
        self.assertEqual([turn.assistant_text for turn in b2], ["model:b:1"])
        self.assertEqual(service.store.syncs, [])
        metrics = self.metrics()
        self.assertEqual(sum(row["status"] == "blocked" for row in metrics), 2)
        self.assertEqual(sum(row["cost"]["total_input_tokens"] for row in metrics), 304)
        merged, delivery = declared_execution_rows({"output": self.temporary.name, "method": "method", "datasets": ["dataset"]})
        self.assertEqual(delivery["dataset"], {"completed": 2, "failed": 1, "blocked": 2})
        self.assertEqual(len(merged), 5)

    async def test_p1_fixed_arrivals_preserved_across_failure(self):
        service = FakeService({"conversation:1"})
        await self.run_queue(source_rows(count=3), "P1", service, fixed=True)
        dispatch = json.loads((self.directory/"workload_dispatch.json").read_text())
        for row in dispatch["actual_arrivals"].values():
            self.assertEqual(row["planned_arrival_seconds"], 0)
            self.assertEqual(row["actual_arrival_seconds"], 0)
            self.assertEqual(row["predecessor_gate"], "terminal_request")
        self.assertEqual(len(service.calls), 3)

    async def test_resume_keeps_each_p1_terminal_failure_without_resubmission(self):
        rows = source_rows()
        service = FakeService({"conversation:1", "conversation:3"})
        await self.run_queue(rows, "P1", service)
        costs = [row["cost"] for row in self.failures()]
        service.calls.clear()
        await self.run_queue(rows, "P1", service)
        self.assertEqual(service.calls, [])
        self.assertEqual([row["cost"] for row in self.failures()], costs)
        self.assertEqual(len(self.metrics()), 4)

    async def test_unresolved_native_session_is_not_silently_abandoned(self):
        service = FakeService({"conversation:1"}, unresolved=True)
        await self.run_queue(source_rows(count=3), "P1", service)
        self.assertEqual(len(service.calls), 1)
        failure = self.failures()[0]
        self.assertEqual(failure["blocked_reason"], "session_cleanup_unresolved")
        self.assertIn("cleanup_error", failure)
        self.assertEqual(failure["blocked_task_ids"], ["conversation:2", "conversation:3"])

    async def test_returning_close_session_with_pending_retirement_still_blocks_p1(self):
        store = SessionStore(self.directory/"state"/"sessions.sqlite3")
        engine = PendingRetirementEngine()
        service = Controller(store, engine)
        PaidFailureQA.calls = []
        await self.run_queue(source_rows(count=3), "P1", service, qa_class=PaidFailureQA)
        failure = self.failures()[0]
        self.assertEqual(PaidFailureQA.calls, ["conversation:1"])
        self.assertEqual(failure["blocked_reason"], "session_cleanup_unresolved")
        self.assertIn("session cleanup unacknowledged", failure["cleanup_error"]["message"])
        self.assertEqual(failure["cost"]["reader_input_tokens"], 3)
        reopened = SessionStore(self.directory/"state"/"sessions.sqlite3")
        try:
            self.assertTrue(reopened.outbox(failure["session_id"]))
        finally:
            reopened.close()
        original_window = failure["request_window"]
        resumed = Controller(SessionStore(self.directory/"state"/"sessions.sqlite3"), engine)
        await self.run_queue(source_rows(count=3), "P1", resumed, qa_class=PaidFailureQA)
        self.assertEqual(PaidFailureQA.calls, ["conversation:1"])
        self.assertEqual(self.failures()[0]["cleanup_recovery"][-1]["status"], "pending")
        engine.retirement_ready = True
        resumed = Controller(SessionStore(self.directory/"state"/"sessions.sqlite3"), engine)
        await self.run_queue(source_rows(count=3), "P1", resumed, qa_class=PaidFailureQA)
        self.assertEqual(PaidFailureQA.calls, ["conversation:1", "conversation:2", "conversation:3"])
        first = self.failures()[0]
        self.assertEqual(first["cleanup_recovery"][-1]["status"], "acknowledged")
        self.assertEqual(first["request_window"], original_window)
        self.assertNotIn("cleanup_error", first)


class DispatcherFailureGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_p1_failure_release_does_not_release_p3_commit_gate(self):
        rows = source_rows(count=2)
        manifest = freeze_workload(rows, name="test", concurrency=1)
        dispatcher = WorkloadDispatcher(manifest)
        await dispatcher.wait_turn(rows[0], ())
        await dispatcher.fail_turn(rows[0]["task_id"], "model failure")
        actual_history_wait = asyncio.create_task(dispatcher.wait_turn(rows[1], ()))
        await asyncio.sleep(0)
        self.assertFalse(actual_history_wait.done())
        actual_history_wait.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await actual_history_wait
        p1 = await asyncio.wait_for(dispatcher.wait_turn(rows[1], (), predecessor_terminal=True), .2)
        self.assertEqual(p1["predecessor_gate"], "terminal_request")
        self.assertNotIn(rows[0]["task_id"], dispatcher._completion_times)


class SessionCleanupReceiptTests(unittest.IsolatedAsyncioTestCase):
    async def test_unacknowledged_demand_release_blocks_reopen_but_ttl_policy_does_not(self):
        with TemporaryDirectory() as temporary:
            store = SessionStore(Path(temporary)/"sessions.sqlite3")
            service = Controller(store, MockEngine())
            try:
                store.read_snapshot("session")
                receipt = CompletionReceipt("r", "a", "engine", 1, "restart", "hash", "stop", True)
                owner = PrefixOwnerRecord("owner", "session", "a", "read", receipt, ConfigFingerprint(),
                    (1, 2), (), "fixed_ttl", {}, 10, stage="RETAINED")
                store.save_prefix_owner(owner, "retained")
                await close_failed_session(service, "session")
                store.reopen_session("session")
                store.save_prefix_owner(PrefixOwnerRecord("demand", "session", "a", "read", receipt,
                    ConfigFingerprint(), (1, 2), (), "joint_demand", {}, 10, stage="RETAINED"), "retained")
                # The actual close path marks a demand owner RELEASING even
                # when its asynchronous release has not acknowledged it yet.
                with self.assertRaisesRegex(RuntimeError, "session cleanup unacknowledged"):
                    await close_failed_session(service, "session")
                self.assertEqual(store.prefix_owner("demand").stage, "RELEASING")
            finally:
                store.close()


class CapacityDemandConfigurationTests(unittest.TestCase):
    def test_next_use_policy_receives_declared_native_pricing_asset(self):
        spec, = full_queue_method_specs(["context_cache_demand"], {}, {},
            demand_policy={"allocation": "next_use", "pricing_asset": "measured-native-prices.json"})
        retention = spec["overrides"]["execution"]["prefix_retention"]
        self.assertEqual(retention["pricing_asset"], str(Path("measured-native-prices.json").resolve()))

    def prepare_fixture(self, directory, *, action_policy=None):
        from controller.action_proposals import finite_action_domain

        root = Path(directory)
        output = root/"prepared"
        output.mkdir()
        run_path, config_path, matrix_path = root/"run.yaml", root/"study.yaml", root/"matrix.yaml"
        tasks_path, asset_path = root/"tasks.jsonl", root/"gain.json"
        run_path.write_text(yaml.safe_dump({"engine": {"kind": "mock"}, "model": {"revision": "reader"}}))
        settings = {"validation_thresholds": {"min_paired_states": 50}}
        if action_policy is not None:
            settings["action_policy"] = action_policy
        config_path.write_text(yaml.safe_dump({"generation": {"run_config": str(run_path)},
            "execution": {"lambda_time": 0, "plan_gain": settings}, "reading": {"output_tokens_per_fact": 64}}))
        matrix_path.write_text(yaml.safe_dump({"protocol": "P1", "resources": {},
            "final_evaluation": {"main_methods": ["context_native"], "acceptance": {}},
            "datasets": {"sample": {"input": str(tasks_path), "primary_metric": "f1"}}}))
        tasks_path.write_text("".join(json.dumps(row)+"\n" for row in source_rows(count=1)))
        asset_path.write_text(json.dumps({"asset_id": "legacy", "reader_checkpoint": "reader", "reader_profile_id": "profile",
            "provenance": {"input_protocol": {"semantics": "obsolete", "reader_fingerprint": {"model_revision": "reader"}},
                "label": "actual_final_quality_minus_actual_direct_answer_quality", "training_split": "train", "training_records": 1,
                "action_domain": finite_action_domain((256,), 64), "continuation_gap_limits": {"max_units": 8, "max_passages": 2}},
            "cost_provenance": {}}))
        return output, SimpleNamespace(config=str(config_path), matrix=str(matrix_path), online_gpu_devices=[],
            split="test", cohorts=None, datasets=None, arrival_plan=None, reader_base_url="http://127.0.0.1:30000",
            methods=["context_native"], gain_asset=str(asset_path), reader_profile_id=None,
            online_process_roots=[os.getpid()], prepare_only=True, lambda_time=None, gap_max_units=None,
            gap_max_passages=None, read_outputs=None, sampling_interval=.01, contextpilot_checkout=None,
            external_reranker="unused", concurrency=[1], repetitions=1, project_root=str(Path(__file__).parents[1]))

    def test_validated_preflight_reads_old_metadata_without_loading_weights(self):
        with TemporaryDirectory() as temporary:
            output, args = self.prepare_fixture(temporary)
            with patch("training.train_plan_gain.PlanGainModel.load", side_effect=AssertionError("must not load old weights")) as load:
                manifest = prepare(args, output)
            self.assertFalse(load.called)
            admission = manifest["action_policy_preflight"]
            self.assertEqual(admission["policy"], "rule")
            self.assertIn("input_protocol_requires_recollection", admission["reasons"])
            method, = yaml.safe_load((output/"study-matrix.yaml").read_text())["methods"]
            settings = method["overrides"]["execution"]["plan_gain"]
            self.assertEqual(settings["output_budgets"], [256])
            self.assertEqual(settings["gap_limits"], {"max_units": 8, "max_passages": 2})
            self.assertEqual(settings["action_policy"], "validated")
            self.assertEqual(settings["validation_thresholds"], {"min_paired_states": 50})

    def test_explicit_learned_preflight_keeps_strict_protocol_validation(self):
        with TemporaryDirectory() as temporary:
            output, args = self.prepare_fixture(temporary, action_policy="learned")
            with patch("training.train_plan_gain.PlanGainModel.load", side_effect=ValueError("strict input protocol mismatch")) as load:
                with self.assertRaisesRegex(ValueError, "strict input protocol mismatch"):
                    prepare(args, output)
            self.assertTrue(load.called)


if __name__ == "__main__":
    unittest.main()
