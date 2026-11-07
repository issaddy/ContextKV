"""Need-specific conditions survive task interpretation and shared reading."""
from dataclasses import replace
import json
import unittest

from adapters.history_state import TaskInterpreter, TaskPrediction
from adapters.support_model import query_view
from controller.context_schema import Condition, Need, RetrievalState, TextRef, VersionIntent
from controller.reading import numeric_read_scopes
from controller.schema import OnlineTurn, TurnRecord


QUESTION = "Count employees in 2023 and sum revenue in 2024."


def reference(text, *, question=QUESTION, turn=1):
    start = question.index(text)
    return TextRef(turn, "user", start, start+len(text))


def initial_prediction():
    conditions = (
        Condition("year23", value_spans=(reference("2023"),), relation_type="TIME"),
        Condition("year24", value_spans=(reference("2024"),), relation_type="TIME"))
    needs = (
        Need("count", (reference("Count employees"),), "count", condition_ids=("year23",)),
        Need("sum", (reference("sum revenue"),), "sum", condition_ids=("year24",)))
    return TaskPrediction("standalone", conditions=conditions, needs=needs)


def interpret(prediction, *, question=QUESTION, turn=1, memory=None):
    class Model:
        def predict(self, *args):
            return prediction

    class HistoryIndex:
        def select(self, *args, **kwargs):
            return ()

    online = OnlineTurn("session", f"task{turn}", turn, "corpus", question, (), (), "P3")
    return TaskInterpreter(Model(), HistoryIndex()).interpret(online, memory or RetrievalState(0))


def prior_memory():
    return RetrievalState(1, (TurnRecord(1, QUESTION, "Recorded actual answer."),),
                          (interpret(initial_prediction()),))


class NeedConditionScopeTests(unittest.TestCase):

    def test_selected_history_supplies_the_object_of_an_elliptical_query(self):
        previous = "Explain escort driver certification."
        question = "When do I need one?"
        antecedent = TextRef(1, "user", 8, len(previous)-1)
        prior = interpret(TaskPrediction("standalone", needs=(Need("certification", (antecedent,)),)),
                          question=previous)
        memory = RetrievalState(1, (TurnRecord(1, previous, "The application fee is $40."),), (prior,))
        prediction = TaskPrediction("followup", resume_task_id=prior.task_id, history_spans=(antecedent,),
            needs=(Need("when", (reference(question, question=question, turn=2),)),))
        state = interpret(prediction, question=question, turn=2, memory=memory)
        self.assertEqual(state.query.splitlines()[0], question)
        self.assertIn("escort driver certification", state.query)
        self.assertEqual(state.working_history, (antecedent,))
        self.assertNotIn("$40", state.query)

    def test_independent_conditions_remain_separate_in_support_and_shared_reading(self):
        state = interpret(initial_prediction())
        self.assertEqual([need.condition_ids for need in state.needs], [("year23",), ("year24",)])
        queries = [json.loads(query_view(state, need, lambda ref: QUESTION[ref.start:ref.end]))
                   for need in state.needs]
        self.assertEqual([[row["values"] for row in query["conditions"]] for query in queries],
                         [[["2023"]], [["2024"]]])
        self.assertEqual([scope.need_ids for scope in numeric_read_scopes(state, ())], [("count",), ("sum",)])
        previous_version = replace(state, version_intent=VersionIntent("as_of",
            document_families=("policy",), version_ids=("v2",), resolved_version_ids=("v1",),
            temporal_relation="before", source_turns=(1,)))
        exhaustive = replace(state.needs[0], scope="all_supplied_documents")
        resolved = json.loads(query_view(previous_version, exhaustive, lambda ref: QUESTION[ref.start:ref.end]))
        self.assertEqual(resolved["scope"], "all_supplied_documents")
        self.assertEqual(resolved["version_intent"]["version_ids"], ["v2"])
        self.assertEqual(resolved["version_intent"]["resolved_version_ids"], ["v1"])
        self.assertEqual(resolved["version_intent"]["temporal_relation"], "before")

    def test_explicit_shared_condition_keeps_one_numeric_scope(self):
        prediction = initial_prediction()
        prediction = replace(prediction, conditions=prediction.conditions[:1], needs=tuple(
            replace(need, condition_ids=("year23",)) for need in prediction.needs))
        state = interpret(prediction)
        scopes = numeric_read_scopes(state, ())
        self.assertEqual([(scope.need_ids, scope.condition_ids) for scope in scopes],
                         [(("count", "sum"), ("year23",))])


    def test_replace_updates_only_needs_linked_to_the_replaced_condition(self):
        question = "Use 2025 for the employee count."
        replacement = Condition("year25", value_spans=(reference("2025", question=question, turn=2),),
            relation_type="TIME", target_refs=("year23",), supersedes=("year23",), update="REPLACE")
        memory = prior_memory()
        for needs in ((), memory.prior_task_states[0].needs):
            with self.subTest(explicit_needs=bool(needs)):
                state = interpret(TaskPrediction("correction", conditions=(replacement,), needs=needs,
                    history_spans=(TextRef(1, "user", 0, len(QUESTION)),)),
                                  question=question, turn=2, memory=memory)
                self.assertEqual([need.condition_ids for need in state.needs], [("year25",), ("year24",)])
                self.assertEqual({condition.condition_id for condition in state.conditions}, {"year25", "year24"})
                self.assertNotIn("2023", state.query)
                self.assertIn("2024", state.query)

    def test_remove_drops_only_the_targeted_binding(self):
        question = "Remove the year restriction from the employee count."
        removal = Condition("remove-year", predicate_span=reference("Remove", question=question, turn=2),
                            target_refs=("year23",), update="REMOVE")
        for retired in (Condition("year23", value_spans=(reference("2023"),)),
                        Condition("year23", key_span=reference("2023")),
                        Condition("year23", predicate_span=reference("2023"))):
            with self.subTest(retired=retired):
                memory = prior_memory()
                previous = replace(memory.prior_task_states[0],
                    conditions=(retired, memory.prior_task_states[0].conditions[1]))
                state = interpret(TaskPrediction("correction", conditions=(removal,),
                    history_spans=(TextRef(1, "user", 0, len(QUESTION)),)),
                    question=question, turn=2, memory=replace(memory, prior_task_states=(previous,)))
                self.assertEqual([need.condition_ids for need in state.needs], [(), ("year24",)])
                self.assertEqual([condition.condition_id for condition in state.conditions], ["year24"])
                self.assertNotIn("2023", state.query)
                self.assertIn("2024", state.query)

    def test_key_only_condition_reaches_compiled_read_and_answer(self):
        from controller.context_schema import ContextPlan, EvidenceSet, ReadTask
        from controller.request_compiler import RequestCompiler
        from controller.schema import SessionSnapshot
        from controller.templates import ByteTemplate

        previous = "For wholesale customers only, list the policy."
        base = interpret(TaskPrediction("standalone", conditions=(
            Condition("wholesale", key_span=reference("wholesale customers only", question=previous)),),
            needs=(Need("n", (reference("list the policy", question=previous),),
                        condition_ids=("wholesale",)),)), question=previous)
        history = (TurnRecord(1, previous, "Policy listed."),)
        snapshot = SessionSnapshot("s", raw_history=history)
        memory = RetrievalState(1, history, (base,))
        compiler = RequestCompiler(ByteTemplate())
        for change, literal in (((), "wholesale customers only"), ((
                Condition("retail", key_span=TextRef(2, "user", 0, 6), target_refs=("wholesale",),
                          update="REPLACE"),), "Retail"), ((
                Condition("remove", predicate_span=TextRef(2, "user", 0, 6), target_refs=("wholesale",),
                          update="REMOVE"),), None)):
            question = "Retail fee?" if change and literal else "What is the fee?"
            state = interpret(TaskPrediction("followup", history_spans=(TextRef(1, "user", 0, len(previous)),),
                              conditions=change), question=question, turn=2, memory=memory)
            plan = ContextPlan("p", state, EvidenceSet("s", ()), read_layout="source_first")
            task = ReadTask("read", "READ", ("n",), input_budget=8192)
            reading, = compiler.compile_read(plan, task, (), (), snapshot, outputs=(512,))
            answer, = compiler.compile_answers(snapshot, question, plan, layouts=("existing",), actions=("REBUILD",))
            for candidate in (reading, answer):
                text = bytes(candidate.input_ids).decode()
                self.assertEqual("wholesale customers only" in text, not change)
                if literal:
                    self.assertIn(f'"key":"{literal}"', text)
                else:
                    self.assertIn('"conditions":[]', text)
            self.assertEqual("wholesale customers only" in state.query, not change)


if __name__ == "__main__":
    unittest.main()
