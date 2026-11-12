"""Paid READ failures preserve costs and never masquerade as source coverage."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from controller.context_schema import (
    CompletionReceipt, ContextEpoch, ContextPlan, EvidenceSet, Need, ReadTask,
    StepRecord, TaskState,
)
from controller.engine_client import EngineError
from controller.execution import ReadingExecutor, RecoverableReadError, step_token_spend
from controller.schema import GenerationResult, SessionSnapshot
from controller.templates import ByteTemplate


class MemoryStepStore:
    def __init__(self):
        self.records = {}

    def execution_step(self, attempt, step):
        return self.records.get((attempt, step))

    def save_execution_step(self, record, stage, generation=None):
        key = (record.attempt_id, record.step_id)
        previous = self.records.get(key)
        self.records[key] = (stage, record, generation if generation is not None
            else previous[2] if previous else None)


class NoGenerationEngine:
    async def generate(self, **kwargs):
        raise AssertionError("a persisted RESULT_READY must not be generated twice")


class ReceiptObserver:
    def observe_executed(self, *args, **kwargs):
        return {"observed": True}


class ReadFailureExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setup_execution(self, *, finish="length", text='{"format_version":5,"facts":[', kind="READ"):
        task = ReadTask("read:0", kind, ("need",), ("u",), output_budget=200,
            output_schema={"properties": {"format_version": {"const": 5}}})
        epoch = ContextEpoch("session", 0, "epoch", 1, engine_restart_id="restart")
        candidate = SimpleNamespace(messages=(), input_ids=(1, 2, 3), input_hash="hash")
        receipt = CompletionReceipt("attempt:read:0", "attempt:read:0", "epoch", 1,
            "restart", "hash", finish, True, actual_prefill_tokens=3, actual_output_tokens=4)
        generation = GenerationResult(text, (4, 5, 6, 7), finish, receipt, elapsed_ms=12.5)
        record = StepRecord("attempt", "read:0", task, epoch, input_ids=candidate.input_ids,
            metrics={"cache_config_hash": "cache"})
        store = MemoryStepStore()
        store.save_execution_step(record, "RESULT_READY", generation)
        engine = NoGenerationEngine()
        observer = ReceiptObserver()
        executor = ReadingExecutor(store, engine, ByteTemplate(), request_compiler=observer)
        plan = ContextPlan("plan", TaskState("task", "question", needs=(Need("need"),)),
            EvidenceSet("evidence", ()))
        return executor, store, engine, candidate, plan, task

    async def test_invalid_structured_output_is_recoverable_and_measured(self):
        for text in ("invalid JSON", "[]"):
            with self.subTest(text=text):
                executor, store, _, candidate, plan, task = self.setup_execution(finish="stop", text=text)
                with self.assertRaises(RecoverableReadError) as caught:
                    await executor._generate_step("attempt", SessionSnapshot("session"), plan, task,
                        (), (), candidate=candidate)
                self.assertEqual("invalid_structured_output", caught.exception.reason)
                stage, record, _ = store.execution_step("attempt", task.step_id)
                self.assertEqual(7, step_token_spend(stage, record))

    async def test_truncated_reduce_is_not_retried_as_a_source_read(self):
        executor, _, _, candidate, plan, task = self.setup_execution(kind="REDUCE")
        with self.assertRaises(EngineError) as caught:
            await executor._generate_step("attempt", SessionSnapshot("session"), plan, task,
                (), (), candidate=candidate)
        self.assertNotIsInstance(caught.exception, RecoverableReadError)
        self.assertEqual("READ_STEP_OUTPUT_TRUNCATED", caught.exception.code)

    async def test_receipt_mismatch_is_not_downgraded_to_recoverable_extraction(self):
        executor, store, _, candidate, plan, task = self.setup_execution()
        stage, record, generation = store.execution_step("attempt", task.step_id)
        generation = replace(generation, receipt=replace(generation.receipt, input_hash="wrong"))
        store.save_execution_step(record, stage, generation)
        with self.assertRaises(EngineError) as caught:
            await executor._generate_step("attempt", SessionSnapshot("session"), plan, task,
                (), (), candidate=candidate)
        self.assertNotIsInstance(caught.exception, RecoverableReadError)
        self.assertEqual("READ_STEP_RECEIPT_MISMATCH", caught.exception.code)


if __name__ == "__main__":
    unittest.main()
