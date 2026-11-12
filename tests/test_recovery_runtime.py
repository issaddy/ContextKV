"""Actual rolling execution over durable state and the in-memory Native engine.

Only generated text and physical-cost observations are controlled; source
compilation, quotes, budget admission, READ parsing, split recovery and terminal
ANSWER selection run through production functions.
"""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from bench.mock_engine import MockEngine
from bench.study_costs import failed_request_cost
from controller.candidates import CandidatePolicy, build_plan_candidates
from controller.context_budget import bind_answer_window, completed_source_coverage
from controller.context_schema import (ContextBudget, ContextPlan, EvidenceBlock, EvidenceKey,
    EvidenceSet, EvidenceUnit, Need, ReadTask, SourceSpan, TaskState)
from controller.cost_model import CostModel
from controller.execution import ReadingExecutor
from controller.plan_policy import CompleteCost, NativePlanningState
from controller.rolling_policy import RollingPlanRuntime, ZERO_AUX
from controller.session_store import SessionStore
from controller.templates import ByteTemplate


class StructuredMockEngine(MockEngine):
    async def generate(self, *, response_schema=None, **kwargs):
        # The real parser checks successful outputs; truncation deliberately
        # models an incomplete constrained response.
        return await super().generate(**kwargs)


class ReceiptNativeObserver:
    """Use exact mock quotes and durable paid token counts; no forecast fitting."""
    async def observe(self, plans, state, snapshot, engine, cost_model, arrived, auxiliary_usage, **kwargs):
        candidates = {candidate.candidate_id: candidate for plan in plans
            for step in (*plan.steps, *plan.route_steps) for candidate in step.variants}
        quotes = await engine.quote_batch(tuple(candidates.values()), engine.fingerprint)
        inputs = sum(len(record.input_ids) for record in state.steps)
        outputs = sum(len(record.output_ids) for record in state.steps)
        paid = CompleteCost((time.monotonic()-arrived)*1000, 0, 0, inputs, outputs)
        native = NativePlanningState("test-reader", "test-profile", snapshot, quotes, cost_model, {}, {}, paid,
            {"native_admission": {"gpu_capacity_tokens": engine.gpu_capacity_tokens,
                "native_max_new_tokens_clip": 4096, "page_size": 1}, "native_quote_ms": 0})
        return native, {}


class ScriptedReader(ReadingExecutor):
    def __init__(self, *args, fail_all=False, stop_after_failure=False, stop_before_failure=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []
        self.fail_all, self.stop_after_failure = fail_all, stop_after_failure
        self.stop_before_failure = stop_before_failure

    async def _generate_step(self, attempt_id, snapshot, plan, task, units, claims, **kwargs):
        self.calls.append(task.step_id)
        # The initial large packet truncates. Successful child calls return
        # valid coverage JSON and never supply target answers or fake facts.
        if task.step_id == "policy:source" or self.fail_all:
            self.engine.response_text = "{"*(task.output_budget+1)
        else:
            self.engine.response_text = json.dumps({"format_version": 4, "claims": [],
                "support": ["n0"], "conflicts": [], "open_needs": []}, separators=(",", ":"))
        return await super()._generate_step(attempt_id, snapshot, plan, task, units, claims, **kwargs)

    async def finish_failed_read(self, record, reason):
        if self.stop_before_failure:
            self.stop_before_failure = False
            raise InterruptedError("simulated process exit with paid RESULT_READY")
        result = await super().finish_failed_read(record, reason)
        if self.stop_after_failure:
            self.stop_after_failure = False
            raise InterruptedError("simulated process exit after durable failure completion")
        return result


class RollingRecoveryRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = TemporaryDirectory()
        self.store = SessionStore(Path(self.temporary.name)/"sessions.sqlite3")
        self.template = ByteTemplate()
        self.engine = StructuredMockEngine(fingerprint=self.template.fingerprint, gpu_capacity_tokens=200000)
        self.snapshot = self.store.read_snapshot("session")
        self.attempt = self.store.begin_attempt("session", "request", "hash", "Read the full source", 0)
        text = "\n".join(f"Record {number}: supplied source fact for line {number}." for number in range(12))
        key = EvidenceKey("corpus", "document", "passage", "v1", "digest")
        span = SourceSpan("span", key, 0, len(text))
        self.unit = EvidenceUnit("source", (span,), text, need_ids=("n",), body_tokens=len(text.encode()))
        budget = ContextBudget(output_tokens=128, read_output_tokens=256, max_steps=20, max_total_tokens=200000)
        self.plan = ContextPlan("plan", TaskState("task", "Read the full source",
            needs=(Need("n", operation="lookup", scope="all_supplied_documents"),), source_scope=(key,)),
            EvidenceSet("set", (self.unit,), source_blocks=(EvidenceBlock(key, "source", "source", text),)),
            budget=budget, reading_mode="coverage_only", coverage_scope="all_supplied_documents",
            coverage_units=(self.unit.unit_id,), steps=(ReadTask("policy:source", "READ", ("n",),
                (self.unit.unit_id,), input_budget=12000, output_budget=256),))
        self.plan = bind_answer_window(self.plan, self.template, self.snapshot)
        self.native = ReceiptNativeObserver()
        self.runtime = RollingPlanRuntime(None, self.native, reader_checkpoint="test-reader",
            reader_profile_id="test-profile", output_budgets=(256,), mode="observed_actions", joint_context=True)
        self.cost_model = CostModel()
        self.arrived = time.monotonic()

    async def asyncTearDown(self):
        await self.engine.aclose()
        self.store.close()
        self.temporary.cleanup()

    def answers(self, snapshot, plans, state, *, representations=("existing",), **kwargs):
        # Before coverage is complete, this fixture exposes only the facts
        # projection. Once read, the coverage-only protocol answers from source.
        representations = ("hybrid",) if completed_source_coverage(
            state.plan, state.results, (self.unit,)) else ("facts",)
        return tuple(candidate for plan in plans for representation in representations
            for candidate in build_plan_candidates(snapshot, plan.task_state.query, plan, self.template,
                outcome=replace(state, plan=plan), representation=representation,
                policy=CandidatePolicy(max_new_tokens=plan.budget.output_tokens),
                actions=("REBUILD",), layouts=("existing",)))

    async def execute(self, reader):
        return await asyncio.wait_for(self.runtime.run(self.attempt.attempt_id, self.snapshot,
            (self.plan,), reader, self.engine, self.template, self.cost_model, self.answers,
            arrived=self.arrived, auxiliary_usage=dict(ZERO_AUX), events=None), timeout=10)

    async def test_truncation_splits_then_reaches_answer_with_original_source_coverage(self):
        reader = ScriptedReader(self.store, self.engine, self.template, request_arrived=self.arrived)
        result = await self.execute(reader)
        self.assertEqual(len(reader.calls), 3)
        self.assertEqual(reader.calls.count("policy:source"), 1)
        self.assertTrue(completed_source_coverage(result.outcome.plan, result.outcome.results, (self.unit,)))
        self.assertFalse(result.outcome.plan.terminal_gaps)
        parent = next(record for record in result.outcome.steps if record.step_id == "policy:source")
        self.assertEqual(parent.result.covered_units, ())
        self.assertEqual(len(parent.output_ids), 256)
        self.assertFalse(parent.metrics["read_recovery_pending"])
        self.assertTrue(result.plan.candidate.feasible)
        self.assertEqual(result.plan.candidate.generation_kind, "ANSWER")
        settled = failed_request_cost(self.store, self.store.get_attempt(self.attempt.attempt_id))
        self.assertEqual(settled["reader_input_tokens"], sum(len(record.input_ids) for record in result.outcome.steps))
        self.assertEqual(settled["reader_output_tokens"], sum(len(record.output_ids) for record in result.outcome.steps))

    async def test_reused_prompt_measurements_preserve_inputs_and_complete_budget(self):
        from controller.execution import require_completion

        reader = ScriptedReader(self.store, self.engine, self.template, request_arrived=self.arrived)
        admitted = []
        plain = ByteTemplate()

        def check_admission(*args, **kwargs):
            result = require_completion(*args, **kwargs)
            self.assertEqual(result, require_completion(*args, **{**kwargs, "template": plain}))
            calls = backend.call_count
            self.assertEqual(result, require_completion(*args, **kwargs))
            self.assertEqual(backend.call_count, calls)
            candidate = args[5]
            rendered = plain.render(candidate.messages)
            self.assertEqual(candidate.input_ids, rendered.input_ids)
            self.assertEqual(candidate.input_hash, rendered.input_hash)
            admitted.append(result)
            return result

        with patch.object(self.template, "render_many", wraps=self.template.render_many) as backend, \
                patch("controller.execution.require_completion", side_effect=check_admission):
            result = await self.execute(reader)
        self.assertEqual(len(admitted), 3)
        self.assertTrue(completed_source_coverage(result.outcome.plan, result.outcome.results, (self.unit,)))

    async def test_recovery_exhaustion_delivers_explicit_gap_without_fabricated_coverage(self):
        reader = ScriptedReader(self.store, self.engine, self.template, request_arrived=self.arrived, fail_all=True)
        result = await self.execute(reader)
        self.assertLessEqual(len(reader.calls), 7)
        self.assertEqual(len(reader.calls), len(set(reader.calls)))
        self.assertTrue(result.outcome.plan.terminal_gaps)
        self.assertFalse(result.outcome.claims)
        self.assertFalse(completed_source_coverage(result.outcome.plan, result.outcome.results, (self.unit,)))
        self.assertEqual(result.plan.candidate.generation_kind, "ANSWER")
        self.assertIn("read_recovery_exhausted", "\n".join(message.content for message in result.plan.candidate.messages))

    async def test_restart_after_paid_failure_does_not_resubmit_the_parent(self):
        reader = ScriptedReader(self.store, self.engine, self.template, request_arrived=self.arrived,
            stop_after_failure=True)
        with self.assertRaises(InterruptedError):
            await self.execute(reader)
        self.assertEqual(reader.calls, ["policy:source"])
        self.assertEqual(self.store.execution_step(self.attempt.attempt_id, "policy:source")[0], "COMPLETE")
        resumed = ScriptedReader(self.store, self.engine, self.template, request_arrived=self.arrived)
        result = await self.execute(resumed)
        self.assertNotIn("policy:source", resumed.calls)
        self.assertEqual(len(resumed.calls), 2)
        self.assertTrue(completed_source_coverage(result.outcome.plan, result.outcome.results, (self.unit,)))
        parent_generations = [event for event in self.engine.events if event["event"] == "generate_start"
            and event["attempt_id"].endswith(":policy:source")]
        self.assertEqual(len(parent_generations), 1)

    async def test_restart_from_result_ready_reuses_the_paid_native_result(self):
        reader = ScriptedReader(self.store, self.engine, self.template, request_arrived=self.arrived,
            stop_before_failure=True)
        with self.assertRaises(InterruptedError):
            await self.execute(reader)
        self.assertEqual(self.store.execution_step(self.attempt.attempt_id, "policy:source")[0], "RESULT_READY")
        resumed = ScriptedReader(self.store, self.engine, self.template, request_arrived=self.arrived)
        result = await self.execute(resumed)
        self.assertTrue(completed_source_coverage(result.outcome.plan, result.outcome.results, (self.unit,)))
        parent_generations = [event for event in self.engine.events if event["event"] == "generate_start"
            and event["attempt_id"].endswith(":policy:source")]
        self.assertEqual(len(parent_generations), 1)
        settled = failed_request_cost(self.store, self.store.get_attempt(self.attempt.attempt_id))
        self.assertEqual(settled["reader_input_tokens"], sum(len(record.input_ids) for record in result.outcome.steps))
        self.assertEqual(settled["reader_output_tokens"], sum(len(record.output_ids) for record in result.outcome.steps))

    async def test_split_obligations_do_not_bypass_complete_step_budget(self):
        self.plan = replace(self.plan, budget=replace(self.plan.budget, max_steps=2))
        reader = ScriptedReader(self.store, self.engine, self.template, request_arrived=self.arrived)
        result = await self.execute(reader)
        self.assertEqual(reader.calls, ["policy:source"])
        self.assertEqual(result.plan.candidate.generation_kind, "ANSWER")
        self.assertTrue(result.outcome.plan.terminal_gaps)
        self.assertIn("continuation_budget_exhausted", result.outcome.plan.terminal_gaps[0].missing_relation)
        self.assertIn("cumulative_step_budget", result.outcome.plan.terminal_gaps[0].missing_relation)


if __name__ == "__main__":
    unittest.main()
