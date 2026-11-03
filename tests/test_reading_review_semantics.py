"""Execute gap closure and finite reduction using the real reading runtime."""
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bench.mock_engine import MockEngine
from controller.claim_graph import ClaimGraph, claim_frontier
from controller.context_schema import (
    ContextBudget, EvidenceClaim, EvidencePool, EvidenceSet, EvidenceUnit, Need,
    OpenNeed, ReadResult, ReductionState, SourceSpan, TaskState,
)
from controller.evidence import make_evidence
from controller.execution import ReadingExecutor
from controller.reading import advance_read_plan, build_read_plan, unresolved_needs
from controller.session_store import SessionStore
from controller.templates import ByteTemplate


def material(name, text):
    block = make_evidence(text, name)
    unit = EvidenceUnit(name, (SourceSpan(name, block.key, 0, len(text)),), text,
                        need_ids=("n",))
    return block, unit


def fact(entity, *, span_ids=(), parent_ids=(), text=None):
    return {"need_id": "n", "text": entity if text is None else text,
            "entity": entity, "relation": "fact", "value": entity, "unit": "",
            "conditions": [], "version": "v1", "span_ids": list(span_ids),
            "parent_claim_ids": list(parent_ids)}


class ReadingEngine(MockEngine):
    def __init__(self, *, compress=True, first_gap=False):
        super().__init__()
        self.compress, self.first_gap = compress, first_gap
        self.calls = []

    async def generate(self, *, response_schema=None, **kwargs):
        rendered = bytes(kwargs["input_ids"]).decode()
        payload = json.loads(rendered.split("<|user|>\n")[-1].split("\n<|end|>")[0])
        self.calls.append(payload)
        gaps = []
        if payload["units"]:
            claims = [fact(unit["unit_id"], span_ids=(unit["source_spans"][0]["span_id"],))
                      for unit in payload["units"]]
            if self.first_gap and len(self.calls) == 1:
                gaps = [{"need_id": "n", "missing_relation": "terminal relation",
                         "bound_entities": ["shared"], "temporal_scope": "current",
                         "parent_claim_ids": []}]
        elif self.compress:
            claims = [fact("combined", parent_ids=tuple(c["claim_id"] for c in payload["claims"]))]
        else:
            claims = [fact(c["entity"], parent_ids=(c["claim_id"],), text=c["text"])
                      for c in payload["claims"]]
        self.response_text = json.dumps({"claims": claims, "support": [] if gaps else ["n"],
                                        "conflicts": [], "open_needs": gaps})
        return await super().generate(**kwargs)


class GapSemantics(unittest.TestCase):
    def test_later_support_closes_old_gap_but_a_later_gap_is_new(self):
        gap = OpenNeed("n", "date", ("A",))
        first = ReadResult("first", open_needs=(gap,))
        solved = ReadResult("second", support=("n",))
        self.assertEqual(unresolved_needs((first, solved)), ())
        self.assertEqual(unresolved_needs((solved, first)), (gap,))

    def test_temporal_scopes_stay_distinct_and_shared_parents_are_retained(self):
        current = OpenNeed("n", "date", ("A",), "current", ("a",))
        previous = replace(current, temporal_scope="2020", parent_claim_ids=("b",))
        repeated = replace(current, parent_claim_ids=("c",))
        pending = unresolved_needs((ReadResult("r1", open_needs=(current, previous)),
                                    ReadResult("r2", open_needs=(repeated,))))
        self.assertEqual(len(pending), 2)
        self.assertEqual(pending[0].parent_claim_ids, ("a", "c"))
        self.assertEqual(pending[1], previous)


class ReadingSemantics(unittest.IsolatedAsyncioTestCase):
    async def execute(self, count, engine, *, extend=None, budget=None):
        pairs = [material(str(index), f"Entity {index} participates.") for index in range(count)]
        blocks, units = zip(*pairs)
        state = TaskState("task", "Infer the complete relationship.",
                          needs=(Need("n", operation="infer"),))
        budget = budget or ContextBudget(window_tokens=10000, read_input_tokens=4000,
                                         read_output_tokens=3000, body_tokens=1024)
        plan = build_read_plan(state, EvidenceSet("selected", units, source_blocks=blocks),
                               "all_supplied_documents", budget, template=ByteTemplate())
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        store = SessionStore(Path(self.temp.name)/"state.db")
        snapshot = store.read_snapshot("session")
        attempt = store.begin_attempt("session", "request", "hash", state.raw_question, 0)
        executor = ReadingExecutor(store, engine, ByteTemplate(), extend_open_needs=extend)
        outcome = await executor.execute_reads(attempt.attempt_id, snapshot, plan)
        return outcome, executor, attempt, snapshot

    async def test_completed_gap_does_not_schedule_another_retrieval(self):
        calls = []

        async def extend(*args):
            calls.append(args)
            return EvidencePool((), ())

        outcome, _, _, _ = await self.execute(8, ReadingEngine(first_gap=True), extend=extend)
        self.assertGreaterEqual(len([step for step in outcome.steps if step.task.kind == "READ"]), 2)
        self.assertEqual(calls, [])
        self.assertEqual(outcome.open_needs, ())

    async def test_compressing_reduction_preserves_all_leaves_and_replays_without_generation(self):
        engine = ReadingEngine()
        outcome, executor, attempt, snapshot = await self.execute(100, engine)
        point = outcome.plan.answer_plan.points[0]
        self.assertEqual(len(point.claim_ids), 1)
        self.assertEqual(len(point.source_spans), 100)
        self.assertEqual(len(ClaimGraph(outcome.claims).leaf_claims(point.claim_ids[0])), 100)
        self.assertEqual(outcome.results[-1].reduction.mode, "merged")
        self.assertEqual(advance_read_plan(outcome.plan, outcome.results), ())
        before = len(engine.calls)
        replay = await executor.execute_reads(attempt.attempt_id, snapshot, outcome.plan)
        self.assertEqual(len(engine.calls), before)
        self.assertEqual(replay.plan.answer_plan, outcome.plan.answer_plan)
        self.assertTrue(all(len(step.input_ids)+step.task.output_budget <= outcome.plan.budget.window_tokens
                            for step in outcome.steps))

    async def test_identity_reduction_finishes_as_a_complete_forest(self):
        # Large JSON output allows a non-compressing reducer to return every fact.
        budget = ContextBudget(window_tokens=22000, read_input_tokens=5000,
                               read_output_tokens=15000, body_tokens=1024)
        outcome, executor, attempt, snapshot = await self.execute(
            100, ReadingEngine(compress=False), budget=budget)
        reductions = [step for step in outcome.steps if step.task.kind == "REDUCE"]
        self.assertGreater(len(reductions), 1)
        self.assertTrue(all(":part:0:" in step.step_id for step in reductions))
        self.assertEqual(outcome.results[-1].reduction.mode, "forest")
        point = outcome.plan.answer_plan.points[0]
        self.assertEqual(len(point.claim_ids), 100)
        self.assertEqual(len(point.source_spans), 100)
        self.assertEqual(advance_read_plan(outcome.plan, outcome.results), ())
        before = len(executor.engine.calls)
        replay = await executor.execute_reads(attempt.attempt_id, snapshot, outcome.plan)
        self.assertEqual(len(executor.engine.calls), before)
        self.assertEqual(replay.plan.answer_plan, outcome.plan.answer_plan)


class ReductionFrontierSemantics(unittest.TestCase):
    def test_new_complete_basis_replaces_old_reduction_without_discarding_independent_roots(self):
        block, unit = material("source", "A B C")
        leaves = tuple(EvidenceClaim(name, "n", "extract", text=name, source_spans=unit.source_spans)
                       for name in ("a", "b", "c"))
        old = EvidenceClaim("old", "n", "derived", text="A and B", parent_claim_ids=("a", "b"))
        left = EvidenceClaim("left", "n", "derived", text="A and B", parent_claim_ids=("a", "b"))
        right = EvidenceClaim("right", "n", "derived", text="C", parent_claim_ids=("c",))
        old_state = ReductionState("reduce:old", ("n",), "infer", ("a", "b"), ("old",), "merged")
        new_state = ReductionState("reduce:new", ("n",), "infer", ("a", "b", "c"),
                                   ("left", "right"), "forest")
        state = TaskState("task", "Compare A B C", needs=(Need("n", operation="infer"),))
        prior = ReadResult("old", claims=(old,), reduction=old_state)
        graph = ClaimGraph((*leaves, old))
        self.assertEqual(set(graph.answer_plan(state, results=(prior,)).points[0].claim_ids), {"old", "c"})
        graph.add(left)
        graph.add(right)
        current = ReadResult("right", claims=(right,), reduction=new_state)
        self.assertEqual(graph.answer_plan(state, results=(prior, current)).points[0].claim_ids,
                         ("left", "right"))
        self.assertIn("old", graph.claims)

    def test_unconsumed_input_is_preserved_next_to_derived_conclusions(self):
        _, unit = material("source", "A B")
        first = EvidenceClaim("a", "n", "extract", source_spans=unit.source_spans)
        second = replace(first, claim_id="b")
        derived = EvidenceClaim("derived", "n", "derived", parent_claim_ids=("a",))
        self.assertEqual(claim_frontier((first, second), (derived,)), (second, derived))


class ReadingRecoverySemantics(unittest.IsolatedAsyncioTestCase):
    async def resume_reduction(self, phase):
        from controller.engine_client import UnknownOutcomeError

        class InterruptedEngine(ReadingEngine):
            def __init__(self):
                super().__init__()
                self.reduction_calls, self.interrupted = 0, False

            async def generate(self, **kwargs):
                result = await super().generate(**kwargs)
                payload = self.calls[-1]
                if payload["claims"]:
                    self.reduction_calls += 1
                    chosen = (self.reduction_calls == 2 if phase == "part" else
                              all(claim["kind"] == "derived" for claim in payload["claims"]))
                    if chosen and not self.interrupted:
                        self.interrupted = True
                        raise UnknownOutcomeError("REDUCTION_RESPONSE_INTERRUPTED", payload["step_id"])
                return result

        blocks, units = zip(*(material(str(i), f"Entity {i} participates.") for i in range(100)))
        state = TaskState("task", "Infer the complete relationship.",
                          needs=(Need("n", operation="infer"),))
        budget = ContextBudget(window_tokens=10000, read_input_tokens=4000,
                               read_output_tokens=3000, body_tokens=1024, max_steps=256)
        plan = build_read_plan(state, EvidenceSet("selected", units, source_blocks=blocks),
                               "all_supplied_documents", budget, template=ByteTemplate())
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp)/"state.db")
            snapshot = store.read_snapshot("session")
            attempt = store.begin_attempt("session", "request", "hash", state.raw_question, 0)
            engine = InterruptedEngine()
            executor = ReadingExecutor(store, engine, ByteTemplate())
            with self.assertRaises(UnknownOutcomeError):
                await executor.execute_reads(attempt.attempt_id, snapshot, plan)
            restored_plan = store.context_plan(attempt.attempt_id)
            outcome = await executor.execute_reads(attempt.attempt_id, snapshot, restored_plan)
            stored = store.execution_steps(attempt.attempt_id)
            self.assertEqual(len(outcome.steps), len({row.step_id for row in outcome.steps}))
            self.assertEqual(len(outcome.steps), len(stored))
            self.assertEqual(len(stored), len(engine.calls))
            spent = sum(len(row.input_ids)+len(row.output_ids) for row in stored)
            self.assertEqual(sum(len(row.input_ids)+len(row.output_ids) for row in outcome.steps), spent)
            self.assertEqual(outcome.plan.online_tokens_spent, spent)
            self.assertEqual(len(outcome.plan.answer_plan.points[0].source_spans), 100)
            self.assertEqual(advance_read_plan(outcome.plan, outcome.results), ())
            store.close()

    async def test_partial_source_reduction_resumes_each_physical_node_once(self):
        await self.resume_reduction("part")

    async def test_partial_bundle_merge_resumes_each_physical_node_once(self):
        await self.resume_reduction("merge")

    async def test_complete_read_finishes_prefix_observation_and_retirement_on_resume(self):
        from controller.engine_client import UnknownOutcomeError

        class SourceFirstEngine(MockEngine):
            def __init__(self):
                super().__init__()
                self.calls = 0

            async def generate(self, *, response_schema=None, **kwargs):
                self.calls += 1
                spans = response_schema["properties"]["claims"]["items"]["properties"]["span_ids"]["items"]["enum"]
                self.response_text = json.dumps({"claims": [fact("42", span_ids=(spans[0],))],
                    "support": ["n"], "conflicts": [], "open_needs": []})
                return await super().generate(**kwargs)

        block, unit = material("source", "The answer is 42.")
        state = TaskState("task", "What is the answer?", needs=(Need("n"),))
        plan = build_read_plan(state, EvidenceSet("selected", (unit,), source_blocks=(block,)),
                               "selected_sources", ContextBudget(), template=ByteTemplate(),
                               read_layout="source_first")
        observed = []

        async def observe_prefix(plan, task, units, rendered, record):
            observed.append(record.step_id)
            if len(observed) == 1:
                raise UnknownOutcomeError("PREFIX_CONTROL_RESPONSE_INTERRUPTED")
            return {"owner_id": "source-owner", "state": "RETAINED"}

        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp)/"state.db")
            snapshot = store.read_snapshot("session")
            attempt = store.begin_attempt("session", "request", "hash", state.raw_question, 0)
            engine = SourceFirstEngine()
            executor = ReadingExecutor(store, engine, ByteTemplate(), observe_prefix=observe_prefix)
            with self.assertRaises(UnknownOutcomeError):
                await executor.execute_reads(attempt.attempt_id, snapshot, plan)
            outcome = await executor.execute_reads(attempt.attempt_id, snapshot, store.context_plan(attempt.attempt_id))
            self.assertEqual(engine.calls, 1)
            self.assertEqual(observed, [plan.steps[0].step_id]*2)
            record = outcome.steps[0]
            self.assertEqual(record.metrics["evidence_prefix"]["owner_id"], "source-owner")
            self.assertEqual(engine._epochs[record.engine_epoch.engine_session_id].state, "RETIRED")
            self.assertEqual(store.outbox(), ())
            store.close()


class PendingFrontierSemantics(unittest.TestCase):
    def test_partial_new_reduction_keeps_the_completed_frontier_and_new_extracts(self):
        _, unit = material("source", "A B C")
        leaves = tuple(EvidenceClaim(name, "n", "extract", text=name, source_spans=unit.source_spans)
                       for name in ("a", "b", "c"))
        old = EvidenceClaim("old", "n", "derived", parent_claim_ids=("a", "b"))
        partial = EvidenceClaim("new-part", "n", "derived", parent_claim_ids=("a", "b", "c"))
        prior = ReadResult("old-step", (old,), reduction=ReductionState(
            "old-logical", ("n",), "infer", ("a", "b"), ("old",), "merged"))
        state = TaskState("task", "Infer A B C", needs=(Need("n", operation="infer"),))
        graph = ClaimGraph((*leaves, old, partial))
        answer = graph.answer_plan(state, results=(prior, ReadResult("new-part", (partial,))))
        self.assertEqual(set(answer.points[0].claim_ids), {"old", "c"})


class LogicalReductionRecoverySemantics(unittest.IsolatedAsyncioTestCase):
    async def resume_completed_reduction(self, count):
        class CompletedReductionInterrupted(Exception):
            pass

        blocks, units = zip(*(material(str(i), f"Entity {i} participates.") for i in range(count)))
        expected_spans = {span.span_id for unit in units for span in unit.source_spans}

        class CompletionEvents:
            def __init__(self):
                self.interrupted_step = None

            def emit(self, name, **fields):
                record = fields["step"]
                supplied = {span.span_id for claim in record.result.claims for span in claim.source_spans}
                if (name == "reading_step_complete" and record.task.kind == "REDUCE"
                        and supplied == expected_spans and self.interrupted_step is None):
                    self.interrupted_step = record
                    raise CompletedReductionInterrupted()

        state = TaskState("task", "Infer the complete relationship.",
                          needs=(Need("n", operation="infer"),))
        budget = ContextBudget(window_tokens=10000, read_input_tokens=4000,
                               read_output_tokens=3000, body_tokens=1024)
        plan = build_read_plan(state, EvidenceSet("selected", units, source_blocks=blocks),
                               "all_supplied_documents", budget, template=ByteTemplate())
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp)/"state.db")
            snapshot = store.read_snapshot("session")
            attempt = store.begin_attempt("session", "request", "hash", state.raw_question, 0)
            engine, events = ReadingEngine(), CompletionEvents()
            executor = ReadingExecutor(store, engine, ByteTemplate(), events=events)
            with self.assertRaises(CompletedReductionInterrupted):
                await executor.execute_reads(attempt.attempt_id, snapshot, plan)

            interrupted = events.interrupted_step
            stage, saved, _ = store.execution_step(attempt.attempt_id, interrupted.step_id)
            self.assertEqual(stage, "COMPLETE")
            self.assertIsNone(saved.result.reduction)
            expected_roots = tuple(claim.claim_id for claim in saved.result.claims)
            self.assertEqual(len(expected_roots), 1)
            reduction_calls = [payload for payload in engine.calls if payload["claims"]]
            if count == 10:
                self.assertEqual(len(reduction_calls), 1)
                self.assertEqual({claim["kind"] for claim in reduction_calls[-1]["claims"]}, {"extract"})
            else:
                self.assertGreater(len(reduction_calls), 1)
                self.assertEqual({claim["kind"] for claim in reduction_calls[-1]["claims"]}, {"derived"})
            calls_before = len(engine.calls)

            outcome = await executor.execute_reads(attempt.attempt_id, snapshot,
                                                   store.context_plan(attempt.attempt_id))
            self.assertEqual(len(engine.calls), calls_before)
            completion, = (result.reduction for result in outcome.results if result.reduction is not None)
            self.assertEqual(len(completion.basis_claim_ids), count)
            self.assertEqual(completion.frontier_claim_ids, expected_roots)
            self.assertEqual(completion.mode, "merged")
            point = outcome.plan.answer_plan.points[0]
            self.assertEqual(point.claim_ids, expected_roots)
            self.assertEqual({span.span_id for span in point.source_spans}, expected_spans)
            self.assertEqual(len(ClaimGraph(outcome.claims).leaf_claims(expected_roots[0])), count)
            self.assertEqual(advance_read_plan(outcome.plan, outcome.results), ())

            stored = store.execution_steps(attempt.attempt_id)
            self.assertEqual(len(outcome.steps), len({record.step_id for record in outcome.steps}))
            self.assertEqual(len(outcome.steps), len(stored))
            self.assertEqual(len(stored), calls_before)
            spent = sum(len(record.input_ids)+len(record.output_ids) for record in stored)
            self.assertEqual(sum(len(record.input_ids)+len(record.output_ids) for record in outcome.steps), spent)
            self.assertEqual(outcome.plan.online_tokens_spent, spent)

            replay = await executor.execute_reads(attempt.attempt_id, snapshot,
                                                  store.context_plan(attempt.attempt_id))
            self.assertEqual(len(engine.calls), calls_before)
            self.assertEqual(replay.plan.answer_plan, outcome.plan.answer_plan)
            self.assertEqual(replay.plan.online_tokens_spent, spent)
            store.close()

    async def test_single_pass_complete_reduction_recovers_logical_marker(self):
        await self.resume_completed_reduction(10)

    async def test_final_merge_complete_recovers_full_basis_and_logical_marker(self):
        await self.resume_completed_reduction(100)


if __name__ == "__main__":
    unittest.main()
