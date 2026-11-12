"""Regression coverage for paid failures and complete declared queues."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from bench.result_records import execution_metric_rows, retain_metrics
from bench.study_costs import failed_request_cost, unsubmitted_cost
from bench.study_metrics import declared_queue_summary
from controller.context_schema import CompletionReceipt, ReadTask, StepRecord
from controller.schema import ContextEpoch, GenerationResult
from controller.schema import ConfigFingerprint
from controller.prefix_retention import PrefixOwnerRecord
from controller.session_store import SessionStore


class PaidFailureAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.store = SessionStore(Path(self.temporary.name)/"sessions.sqlite3")
        self.store.read_snapshot("session")
        self.attempt = self.store.begin_attempt("session", "request", "hash", "question", 0)

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def native(self, step_id, *, kind="READ", input_tokens=100, output_tokens=10,
               stage="COMPLETE", elapsed=20, receipt=True, generation=True):
        epoch = ContextEpoch("session", 0, "engine:"+step_id, 1, engine_restart_id="restart")
        native = (CompletionReceipt(step_id, step_id, epoch.engine_session_id, 1, "restart", "hash",
            "length" if stage == "RESULT_READY" else "stop", True,
            actual_device_hit_tokens=20, actual_prefill_tokens=input_tokens-20,
            actual_output_tokens=output_tokens) if receipt else None)
        result = (GenerationResult("answer", tuple(range(output_tokens)), "stop", native,
            elapsed_ms=elapsed) if generation and native is not None else None)
        record = StepRecord(self.attempt.attempt_id, step_id, ReadTask(step_id, kind, ()), epoch,
            input_ids=tuple(range(input_tokens)), output_ids=result.output_ids if result is not None else (),
            receipt=native, metrics={"elapsed_ms": elapsed} if result is not None else {})
        self.store.save_execution_step(record, stage, result)
        return record, result

    def auxiliary(self, identity, input_tokens, output_tokens, *, group_id=None, compute_ms=7):
        value = {"call_id": identity, "attempt_id": self.attempt.attempt_id,
            "kind": "provider_usage" if group_id is None else "task_model",
            "status": "RECORDED", "input_tokens": input_tokens, "output_tokens": output_tokens,
            "compute_ms": compute_ms, "elapsed_ms": compute_ms}
        if group_id is not None:
            value["group_id"] = group_id
        self.store.save_auxiliary_call(value)

    def test_paid_truncation_and_auxiliary_groups_survive_without_response(self):
        self.native("read-one")
        self.native("truncated", input_tokens=80, output_tokens=25, stage="RESULT_READY")
        self.auxiliary("retrieval", 1000, 60)
        self.auxiliary("child", 1000, 60, group_id="retrieval")
        self.auxiliary("gap", 200, 10)
        # This aggregate overlaps both captured provider groups.
        attempt = replace(self.attempt, stage="FAILED", failed_from_stage="STARTED",
            auxiliary_usage={"input_tokens": 1200, "output_tokens": 70, "compute_ms": 14})
        cost = failed_request_cost(self.store, attempt)
        self.assertEqual(cost["reader_input_tokens"], 180)
        self.assertEqual(cost["reader_output_tokens"], 35)
        self.assertEqual(cost["total_input_tokens"], 1380)
        self.assertEqual(cost["total_output_tokens"], 105)
        self.assertEqual(cost["actual_prefill_tokens"], 140)
        self.assertEqual(cost["online_compute_ms"], 54)
        self.assertEqual(cost["accounting"]["unknown_fields"], [])

    def test_rolling_final_answer_is_counted_once(self):
        self.native("read")
        record, generation = self.native("answer", kind="ANSWER", input_tokens=50, output_tokens=3)
        attempt = replace(self.attempt, stage="COMMITTED", result=generation, final_step_id="answer",
            plan=SimpleNamespace(candidate=SimpleNamespace(input_ids=record.input_ids)))
        cost = failed_request_cost(self.store, attempt)
        self.assertEqual(cost["reader_input_tokens"], 150)
        self.assertEqual(cost["reader_output_tokens"], 13)
        self.assertEqual(len(cost["accounting"]["native_call_ids"]), 2)

    def test_terminal_receipt_output_count_used_when_result_is_unavailable(self):
        self.native("cancelled", stage="FAILED", output_tokens=17, generation=False)
        cost = failed_request_cost(self.store, self.attempt)
        self.assertEqual(cost["reader_output_tokens"], 17)
        self.assertEqual(cost["reader_input_tokens"], 100)
        self.assertIsNone(cost["online_compute_ms"])

    def test_unknown_submitted_work_keeps_known_receipt_lower_bound(self):
        self.native("known", output_tokens=9)
        self.native("unresolved", stage="SUBMITTED", receipt=False, generation=False)
        cost = failed_request_cost(self.store, self.attempt)
        self.assertIsNone(cost["reader_input_tokens"])
        self.assertIsNone(cost["total_output_tokens"])
        self.assertIsNone(cost["actual_prefill_tokens"])
        self.assertEqual(cost["accounting"]["known_lower_bound"]["reader_input_tokens"], 100)
        self.assertEqual(cost["accounting"]["known_lower_bound"]["total_output_tokens"], 9)

    def test_unknown_provider_aggregate_does_not_hide_known_children(self):
        self.auxiliary("unfinished", None, None, compute_ms=None)
        self.auxiliary("paid", 30, 4, group_id="unfinished", compute_ms=2)
        cost = failed_request_cost(self.store, self.attempt)
        self.assertIsNone(cost["auxiliary_input_tokens"])
        self.assertEqual(cost["accounting"]["known_lower_bound"]["auxiliary_input_tokens"], 30)
        self.assertEqual(cost["accounting"]["known_lower_bound"]["auxiliary_output_tokens"], 4)

    def test_boolean_usage_is_unknown_instead_of_one_token(self):
        self.auxiliary("invalid-usage", True, False)
        cost = failed_request_cost(self.store, self.attempt)
        self.assertIsNone(cost["total_input_tokens"])
        self.assertIsNone(cost["total_output_tokens"])
        self.assertEqual(cost["accounting"]["known_lower_bound"]["total_input_tokens"], 0)

    def test_partial_native_output_is_a_lower_bound_until_final_receipt(self):
        record, generation = self.native("partial", stage="RESULT_READY", output_tokens=5)
        receipt = replace(record.receipt, generation_complete=False, actual_output_tokens=None)
        self.store.save_execution_step(replace(record, receipt=receipt), "RESULT_READY",
            replace(generation, generation_complete=False, receipt=receipt))
        cost = failed_request_cost(self.store, self.attempt)
        self.assertIsNone(cost["reader_output_tokens"])
        self.assertEqual(cost["accounting"]["known_lower_bound"]["reader_output_tokens"], 5)

    def test_unsubmitted_candidates_are_not_charged(self):
        self.native("prepared", stage="PREPARED", receipt=False, generation=False)
        self.store.save_execution_step(StepRecord(self.attempt.attempt_id, "cpu", ReadTask("cpu", "REDUCE", ()),
            metrics={"cpu_reduce_ms": 5}), "COMPLETE")
        cost = failed_request_cost(self.store, self.attempt)
        self.assertEqual(cost["reader_input_tokens"], 0)
        self.assertEqual(cost["total_output_tokens"], 0)
        self.assertEqual(cost["online_compute_ms"], 0)

    def test_submitted_final_answer_without_receipt_is_unknown(self):
        cost = failed_request_cost(self.store, replace(self.attempt, stage="FAILED", failed_from_stage="SUBMITTED"))
        self.assertIsNone(cost["total_input_tokens"])
        self.assertEqual(cost["accounting"]["native_call_ids"], ["answer"])


class DeclaredTaskRecordsTests(unittest.TestCase):
    def test_failed_and_blocked_costs_survive_metrics_cleanup(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"state").mkdir()
            manifest = {"method": {"id": "context"}, "dataset": "multi", "split": "test", "protocol": "P3",
                "scheduled_task_ids": ["one", "two", "three"]}
            (root/"generation_manifest.json").write_text(json.dumps(manifest))
            cost = {**unsubmitted_cost(), "total_input_tokens": 120, "total_output_tokens": 10, "complete_ms": 20}
            success = {"task_id": "one", "method": "context", "dataset": "multi", "split": "test",
                "protocol": "P3", "status": "completed", "metrics": {"f1": .8}, "cost": cost,
                "answer": "private answer", "scoring": {"status": "scored"}}
            failure = {"task_id": "two", "conversation_id": "conv", "status": "failed",
                "cost": {**cost, "total_input_tokens": 40}, "blocked_task_ids": ["three"],
                "blocked_tasks": [{"task_id": "three", "turn": 3}]}
            (root/"failed_conversations.json").write_text(json.dumps([failure]))
            retain_metrics(root, [success])
            rows = [json.loads(line) for line in (root/"task_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["status"] for row in rows], ["completed", "failed", "blocked"])
            self.assertNotIn("answer", rows[0])
            self.assertFalse((root/"state").exists())
            self.assertEqual(rows[2]["cost"]["total_input_tokens"], 0)
            self.assertIsNone(rows[2]["cost"]["complete_ms"])
            self.assertEqual(rows[2]["blocked_by_task_id"], "two")
            summary = declared_queue_summary(rows, {"primary_metric": "f1"}, ["one", "two", "three"])
            self.assertAlmostEqual(summary["queue_primary_score"], .8/3)
            self.assertEqual(summary["cost"]["total_input_tokens"], 160)
            self.assertIsNone(summary["cost"]["p95_complete_ms"])

    def test_unknown_failure_cost_retains_queue_lower_bound_and_identity(self):
        cost = {"total_input_tokens": None, "accounting": {"known_lower_bound": {"total_input_tokens": 80}}}
        rows = [{"task_id": "failed", "status": "failed", "cost": cost, "metrics": {}},
                {"task_id": "blocked", "status": "blocked", "cost": unsubmitted_cost(), "metrics": {}}]
        summary = declared_queue_summary(rows, {"primary_metric": "f1"}, ["failed", "blocked"])
        self.assertIsNone(summary["cost"]["total_input_tokens"])
        self.assertEqual(summary["cost"]["accounting"]["total_input_tokens"],
            {"known_lower_bound": 80, "unknown_task_ids": ["failed"]})

    def test_successful_resume_wins_over_previous_failure(self):
        with TemporaryDirectory() as directory:
            rows = execution_metric_rows(directory, [{"task_id": "t", "status": "completed", "metrics": {}}],
                failures=[{"task_id": "t", "blocked_task_ids": ["tail"]}])
            self.assertEqual([row["task_id"] for row in rows], ["t"])

    def test_missing_execution_keeps_queue_total_unknown(self):
        rows = [{"task_id": "known", "status": "failed", "cost": {"total_input_tokens": 50}, "metrics": {}}]
        summary = declared_queue_summary(rows, {"primary_metric": "f1"}, ["known", "missing"])
        self.assertIsNone(summary["cost"]["total_input_tokens"])
        self.assertEqual(summary["cost"]["accounting"]["total_input_tokens"],
                         {"known_lower_bound": 50, "unknown_task_ids": ["missing"]})

    def test_unresolved_native_state_is_preserved_after_scoring_cleanup(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"state").mkdir()
            (root/"failed_conversations.json").write_text(json.dumps([
                {"task_id": "unresolved", "cleanup_error": {"message": "still running"}}]))
            retain_metrics(root, [])
            self.assertTrue((root/"state").is_dir())

    def test_successful_scoring_does_not_delete_unacknowledged_native_retirement(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SessionStore(root/"state"/"sessions.sqlite3")
            store.read_snapshot("session")
            attempt = store.begin_attempt("session", "request", "hash", "q", 0)
            epoch = ContextEpoch("session", 0, "native", 1, engine_restart_id="restart")
            receipt = CompletionReceipt("native-call", "native-call", "native", 1, "restart", "hash", "stop", True)
            generation = GenerationResult("result", (4,), "stop", receipt)
            store.save_execution_step(StepRecord(attempt.attempt_id, "read", ReadTask("read", "READ", ()),
                epoch, input_ids=(1, 2), output_ids=(4,), receipt=receipt), "COMPLETE", generation)
            store.close()
            retain_metrics(root, [{"task_id": "success", "status": "completed", "metrics": {"f1": 1}}])
            self.assertTrue((root/"state").is_dir())
            store = SessionStore(root/"state"/"sessions.sqlite3")
            for operation in store.outbox():
                store.ack_retire(operation.operation_id)
            store.close()
            retain_metrics(root, [{"task_id": "success", "status": "completed", "metrics": {"f1": 1}}])
            self.assertFalse((root/"state").exists())

    def test_successful_scoring_preserves_active_demand_owner_state(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SessionStore(root/"state"/"sessions.sqlite3")
            receipt = CompletionReceipt("native-call", "native-call", "native", 1, "restart", "hash", "stop", True)
            store.save_prefix_owner(PrefixOwnerRecord("owner", "session", "a", "read", receipt,
                ConfigFingerprint(), (1, 2), (), "joint_demand", {}, 10, stage="RELEASING"), "release_requested")
            store.close()
            retain_metrics(root, [{"task_id": "success", "status": "completed", "metrics": {"f1": 1}}])
            self.assertTrue((root/"state").is_dir())


if __name__ == "__main__":
    unittest.main()
