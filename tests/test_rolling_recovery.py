import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from controller.context_schema import (AnswerPlan, AnswerPoint, AnswerWindowAllocation,
    ContextPlan, EvidenceBlock, EvidenceClaim, EvidenceKey, EvidenceSet, EvidenceUnit, Message, Need,
    OpenNeed, ReadResult, ReadTask, ReadingOutcome, SourceSpan, StepRecord, TaskState)
from controller.context_budget import (answer_evidence_gaps, completed_source_coverage, completion_reserve,
                                      incomplete_full_coverage)
from controller.plan_policy import PlanCandidate, PlanStep
from controller.read_recovery import (MAX_FIELD_REPAIRS, _replace_records, reconcile_recovery,
    recover_failed_reads, schedule_field_repair, terminalize)
from controller.render_memo import TurnTemplateMemo
from controller.rolling_policy import RollingPlanRuntime, compile_route_steps
from controller.templates import ByteTemplate


def fixture(*, text=None, full=True):
    text = text or "\n".join(f"公司{i}: 流动资产合计 {i+1} 万元；总资产 {i+100} 万元。" for i in range(10))
    key = EvidenceKey("c", "d", "p", "2024", "hash")
    span = SourceSpan("s", key, 0, len(text))
    unit = EvidenceUnit("u", (span,), text, body_tokens=len(text.encode()), need_ids=("n",))
    need = Need("n", operation="sum", scope="all_supplied_documents" if full else "selected_sources")
    task = ReadTask("policy:read", "READ", ("n",), ("u",), operation="sum")
    plan = ContextPlan("p", TaskState("t", "统计流动资产合计", needs=(need,), source_scope=(key,)),
        EvidenceSet("e", (unit,), source_blocks=(EvidenceBlock(key, "test", "test", text),)),
        coverage_scope="all_supplied_documents" if full else "selected_sources", steps=(task,),
        coverage_units=("u",), answer_allocation=AnswerWindowAllocation(1, 1024, 2048, (("n", 1024),)))
    return plan, task, unit


def failed_state():
    plan, task, unit = fixture()
    result = ReadResult(task.step_id, open_needs=(OpenNeed("n", "READ recovery required: truncated_output"),),
                        input_tokens=50, output_tokens=20)
    record = StepRecord("a", task.step_id, task, input_ids=tuple(range(50)), output_ids=tuple(range(20)),
        result=result, metrics={"read_failure": "truncated_output", "read_recovery_pending": True})
    return ReadingOutcome(plan, (result,), (record,), open_needs=result.open_needs), unit


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.template = ByteTemplate()

    def test_split_retains_paid_parent_and_exact_source_geometry(self):
        state, unit = failed_state()
        updated = recover_failed_reads(state, self.template.tokenizer)
        children = updated.steps[0].metrics["read_recovery_children"]
        self.assertEqual(len(children), 2)
        self.assertEqual(updated.steps[0].input_ids, state.steps[0].input_ids)
        self.assertEqual(updated.steps[0].output_ids, state.steps[0].output_ids)
        self.assertEqual(updated.results[0].covered_units, ())
        child_tasks = tuple(task for task in updated.plan.steps if task.step_id in children)
        self.assertTrue(all(task.predecessor_step_ids == (state.steps[0].step_id,) for task in child_tasks))
        by_id = {item.unit_id: item for item in updated.plan.selected_set.selected_units}
        spans = sorted((span.start, span.end) for task in child_tasks
            for identity in task.input_unit_ids for span in by_id[identity].source_spans)
        self.assertEqual(spans, [(0, spans[0][1]), (spans[0][1], len(unit.text))])
        self.assertTrue(incomplete_full_coverage(updated.plan, updated.results))
        self.assertEqual(recover_failed_reads(updated, self.template.tokenizer).plan.steps, updated.plan.steps)

    def test_split_finishes_only_after_all_descendant_coverage(self):
        state, unit = failed_state()
        state = recover_failed_reads(state, self.template.tokenizer)
        children = tuple(task for task in state.plan.steps if task.step_id != state.steps[0].step_id)
        first = StepRecord("a", children[0].step_id, children[0],
            result=ReadResult(children[0].step_id, covered_units=children[0].input_unit_ids))
        state = reconcile_recovery(_replace_records(state, (first,)))
        self.assertTrue(state.open_needs)
        self.assertFalse(completed_source_coverage(state.plan, state.results, (unit,)))
        second = StepRecord("a", children[1].step_id, children[1],
            result=ReadResult(children[1].step_id, covered_units=children[1].input_unit_ids))
        state = reconcile_recovery(_replace_records(state, (second,)))
        self.assertFalse(state.open_needs)
        self.assertTrue(completed_source_coverage(state.plan, state.results, (unit,)))
        self.assertFalse(state.results[0].covered_units)

    def test_unrelated_coverage_does_not_complete_failed_parent(self):
        state, _ = failed_state()
        state = recover_failed_reads(state, self.template.tokenizer)
        unrelated = ReadTask("other", "READ", ("other_need",), ("u",))
        record = StepRecord("a", "other", unrelated, result=ReadResult("other", covered_units=("u",)))
        state = reconcile_recovery(_replace_records(state, (record,)))
        self.assertTrue(state.steps[0].metrics["read_recovery_pending"])
        self.assertTrue(state.open_needs)

    def test_unsplittable_failure_becomes_explicit_terminal_gap(self):
        state, _ = failed_state()
        plan, task, _ = fixture(text="流动资产合计缺失。")
        record = replace(state.steps[0], task=task)
        state = replace(state, plan=plan, steps=(record,))
        state = recover_failed_reads(state, self.template.tokenizer)
        self.assertTrue(state.plan.terminal_gaps)
        self.assertEqual(answer_evidence_gaps(state), {"n": "terminal_evidence_gap"})
        self.assertFalse(answer_evidence_gaps(state, require_complete_evidence=False))
        self.assertTrue(incomplete_full_coverage(state.plan, state.results))
        self.assertFalse(incomplete_full_coverage(state.plan, state.results, allow_terminal_gaps=True))

    def test_split_does_not_cut_an_unseparated_atomic_record(self):
        state, _ = failed_state()
        plan, task, _ = fixture(text="单一字段无分隔内容"*100)
        state = replace(state, plan=plan, steps=(replace(state.steps[0], task=task),))
        state = recover_failed_reads(state, self.template.tokenizer)
        self.assertEqual(len(state.plan.steps), 1)
        self.assertTrue(state.plan.terminal_gaps)

    def test_action_hole_schedules_local_field_repair(self):
        plan, task, _ = fixture()
        gap = OpenNeed("n", "流动资产合计", ("公司0",), "2024")
        result = ReadResult(task.step_id, covered_units=("u",), open_needs=(gap,))
        record = StepRecord("a", task.step_id, task, result=result)
        state = ReadingOutcome(plan, (result,), (record,), open_needs=(gap,))
        state = schedule_field_repair(state, self.template.tokenizer)
        repair = state.plan.steps[-1]
        self.assertEqual(repair.operation, "repair_extract")
        self.assertEqual(repair.repair_gaps, (gap,))
        runtime = object.__new__(RollingPlanRuntime)
        self.assertEqual(runtime._required_scope_step(state).step_id, repair.step_id)
        self.assertTrue(answer_evidence_gaps(state, require_complete_evidence=False))

    def test_field_repair_closes_only_matching_entity_and_period(self):
        plan, task, unit = fixture()
        repaired = OpenNeed("n", "流动资产合计", ("公司0",), "2024")
        remaining = replace(repaired, bound_entities=("公司1",))
        result = ReadResult(task.step_id, covered_units=("u",), open_needs=(repaired, remaining))
        state = ReadingOutcome(plan, (result,), (StepRecord("a", task.step_id, task, result=result),),
                               open_needs=(repaired, remaining))
        state = schedule_field_repair(state, self.template.tokenizer)
        repair = state.plan.steps[-1]
        claim = EvidenceClaim("c", "n", "extract", value="1", unit="万元", entity="公司0",
                              relation="流动资产合计", version="2024", source_spans=unit.source_spans)
        record = StepRecord("a", repair.step_id, repair,
            result=ReadResult(repair.step_id, claims=(claim,), covered_units=repair.input_unit_ids))
        state = reconcile_recovery(_replace_records(state, (record,)))
        self.assertEqual(state.open_needs, (remaining,))

    def test_repair_limit_preserves_missing_fact_and_never_emits_zero(self):
        plan, task, _ = fixture()
        gap = OpenNeed("n", "流动资产合计", ("公司0",), "2024")
        result = ReadResult(task.step_id, covered_units=("u",), open_needs=(gap,))
        state = ReadingOutcome(plan, (result,), (StepRecord("a", task.step_id, task, result=result),), open_needs=(gap,))
        for _ in range(MAX_FIELD_REPAIRS):
            state = schedule_field_repair(state, self.template.tokenizer)
            repair = state.plan.steps[-1]
            state = _replace_records(state, (StepRecord("a", repair.step_id, repair,
                result=ReadResult(repair.step_id, covered_units=repair.input_unit_ids, open_needs=(gap,))),))
        state = schedule_field_repair(state, self.template.tokenizer)
        self.assertTrue(state.plan.terminal_gaps)
        self.assertFalse(state.claims)
        self.assertTrue(state.open_needs)
        self.assertEqual(len([task for task in state.plan.steps if task.operation == "repair_extract"]), MAX_FIELD_REPAIRS)

    def test_terminal_answer_preserves_all_gaps_and_hides_incomplete_aggregate(self):
        from controller.candidates import build_plan_candidates
        from controller.schema import SessionSnapshot

        plan, _, unit = fixture()
        partial = EvidenceClaim("partial", "n", "derived", value="987654321", unit="万元",
                                source_spans=unit.source_spans, operation="sum")
        plan = replace(plan, answer_plan=AnswerPlan((AnswerPoint("n", ("partial",), unit.source_spans),)))
        state = ReadingOutcome(plan, claims=(partial,))
        gaps = (OpenNeed("n", "流动资产合计", ("甲公司",), "2023"),
                OpenNeed("n", "流动资产合计", ("乙公司",), "2024"))
        state = terminalize(state, gaps, "field_recovery_exhausted")
        self.assertEqual(len(state.plan.terminal_gaps), 2)
        snapshot = SessionSnapshot("s", archive=plan.selected_set.source_blocks)
        candidate, = build_plan_candidates(snapshot, plan.task_state.raw_question, state.plan,
            self.template, outcome=state, layouts=("existing",), actions=("REBUILD",), representation="facts")
        prompt = "\n".join(message.content for message in candidate.messages)
        self.assertNotIn("987654321", prompt)
        self.assertIn("甲公司", prompt)
        self.assertIn("乙公司", prompt)
        self.assertIn("Do not infer a missing value", prompt)
        self.assertEqual(answer_evidence_gaps(state), {"n": "terminal_evidence_gap"})
        self.assertFalse(answer_evidence_gaps(state, candidate, require_complete_evidence=False))

    def test_numeric_need_does_not_force_unrelated_read(self):
        plan, task, unit = fixture(full=False)
        other = ReadTask("lookup", "READ", ("lookup",), ("other",))
        need = Need("lookup")
        plan = replace(plan, task_state=replace(plan.task_state, needs=(*plan.task_state.needs, need)),
            steps=(task, other), selected_set=replace(plan.selected_set,
                selected_units=(unit, replace(unit, unit_id="other", need_ids=("lookup",)))))
        claim = EvidenceClaim("n1", "n", "extract", value="100", unit="万元", entity="公司0",
                              relation="流动资产合计", source_spans=unit.source_spans)
        result = ReadResult(task.step_id, claims=(claim,), covered_units=("u",))
        state = ReadingOutcome(plan, (result,), claims=(claim,))
        runtime = object.__new__(RollingPlanRuntime)
        selected = runtime._required_scope_step(state)
        self.assertEqual(selected.kind, "REDUCE")
        self.assertEqual(selected.need_ids, ("n",))
        reserve = completion_reserve(plan, (result,), selected)
        self.assertEqual(reserve.future_input_tokens, 1)


class CompilationTests(unittest.TestCase):
    def test_future_quotes_are_linear_and_full_route_is_retained(self):
        plan, task, unit = fixture()
        tasks = tuple(replace(task, step_id=f"read:{index}") for index in range(50))
        plan = replace(plan, steps=tasks)
        total = 0
        with patch("controller.rolling_policy.compile_step", return_value=(SimpleNamespace(feasible=True),)) as compile_read, \
             patch("controller.action_proposals.observed_step_outputs", return_value=(1, (2048,))):
            for index, current in enumerate(tasks):
                state = ReadingOutcome(plan, results=tuple(ReadResult(task.step_id) for task in tasks[:index]))
                proposal = PlanCandidate("p", "scope_scan", plan,
                    (PlanStep(current, (SimpleNamespace(feasible=True),)),), "reader")
                route = compile_route_steps(proposal, state, None, ByteTemplate(), output_budgets=(2048,),
                                            quote_lookahead=2)
                self.assertEqual(len(route), 50-index)
                quoted = sum(bool(step.variants) for step in route)
                self.assertLessEqual(quoted, 3)
                total += quoted
            self.assertEqual(compile_read.call_count, 97)
        self.assertEqual(total, 147)  # 1,275 quoted READs before the bounded horizon.

    def test_zero_horizon_keeps_current_but_does_not_compile_future(self):
        plan, task, _ = fixture()
        future = replace(task, step_id="future")
        plan = replace(plan, steps=(task, future))
        proposal = PlanCandidate("p", "read_once", plan, (PlanStep(task, (object(),)),), "reader")
        with patch("controller.rolling_policy.compile_step") as compile_read:
            route = compile_route_steps(proposal, ReadingOutcome(plan), None, ByteTemplate(),
                                        output_budgets=(2048,), quote_lookahead=0)
        self.assertFalse(compile_read.called)
        self.assertEqual(len(route), 2)
        self.assertFalse(route[1].variants)

    def test_length_memo_avoids_retokenizing_evicted_prompts(self):
        class CountingTemplate(ByteTemplate):
            count = 0

            def render_many(self, conversations, *, with_offsets=False):
                self.count += len(conversations)
                return super().render_many(conversations, with_offsets=with_offsets)

        source = CountingTemplate()
        memo = TurnTemplateMemo(source, max_prompts=1)
        messages = tuple((Message("user", f"source {index}"),) for index in range(20))
        expected = tuple(len(prompt.input_ids) for prompt in ByteTemplate().render_many(messages))
        self.assertEqual(memo.token_lengths(messages), expected)
        self.assertEqual(memo.token_lengths(tuple(reversed(messages))), tuple(reversed(expected)))
        self.assertEqual(source.count, 20)
        self.assertEqual(len(memo.render_many(messages)), 20)
        self.assertLessEqual(len(memo._prompts), 1)
        # An offset request never reuses a non-offset rendered object.
        self.assertIsNotNone(memo.render(messages[-1], with_offsets=True).offsets)


if __name__ == "__main__":
    unittest.main()
