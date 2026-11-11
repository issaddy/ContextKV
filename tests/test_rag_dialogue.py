"""Conventional QA wiring: visible sources, actual history, one answer call."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from bench.mock_engine import MockEngine
from controller.candidates import CandidatePolicy
from controller.context_schema import ContextBudget
from controller.evidence import make_evidence
from controller.qa import DocumentQA
from controller.service import Controller, EvidenceSelection
from controller.session_store import SessionStore


class RAGEngine(MockEngine):
    def __init__(self):
        super().__init__()
        self.inputs = []

    async def generate(self, **kwargs):
        text = bytes(kwargs["input_ids"]).decode()
        self.inputs.append(text)
        self.response_text = ("Lumen application next steps" if "Rewrite the latest question" in text
                              else "Use form W-7.")
        return await super().generate(**kwargs)


class OrdinaryQATest(unittest.IsolatedAsyncioTestCase):
    async def test_packing_history_and_single_answer_share_the_normal_ledger(self):
        body = "Use form W-7. "+"Keep the submission receipt. "*300
        block = replace(make_evidence(body, "guide"), span=(137, 137+len(body)))

        class Provider:
            corpus_snapshot_id = "docs"
            budget = ContextBudget(window_tokens=4096, output_tokens=128, history_tokens=512)

            def __init__(self):
                self.queries = []

            async def retrieve(self, snapshot, query, task_id):
                self.queries.append((query, snapshot.raw_history))
                return EvidenceSelection((block, block), (block.key,), auxiliary_usage={
                    "input_tokens": 7, "output_tokens": 0, "compute_ms": 1., "models": []})

            def finish_request(self, session_id, task_id):
                pass

        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory)/"state.db")
            engine, provider = RAGEngine(), Provider()
            controller = Controller(store, engine, dialogue_mode="rag",
                policy=CandidatePolicy(max_total_model_tokens=4096, max_new_tokens=128))
            qa = DocumentQA(controller, provider)
            first = await qa.ask("session", "How do I apply?", "one")
            self.assertEqual(len(engine.inputs), 1)
            step, = store.execution_steps(first.attempt_id)
            self.assertEqual(step.task.kind, "RAG_ANSWER")
            self.assertLessEqual(len(step.input_ids)+step.task.output_budget, 4096)
            spans = step.metrics["source_view"]["evidence_spans"]
            self.assertEqual(len(spans), 1)
            span = spans[0]
            self.assertEqual(span["source_start"], 137)
            self.assertLess(span["source_end"], 137+len(body))
            shown = step.input_messages[span["message_index"]].content[span["start"]:span["end"]]
            self.assertEqual(shown, body[:span["source_end"]-137])
            second = await qa.ask("session", "And then?", "two")
            self.assertEqual(len(engine.inputs), 3)
            self.assertEqual(provider.queries[-1][0], "Lumen application next steps")
            self.assertEqual(provider.queries[-1][1][0].assistant_text, first.assistant_text)
            self.assertEqual([step.task.kind for step in store.execution_steps(second.attempt_id)],
                             ["RAG_CONDENSE", "RAG_ANSWER"])
            self.assertEqual(store.get_snapshot("session").version, 2)
            self.assertEqual(second.total_input_tokens, second.reader_input_tokens+7)
            await controller.aclose()
            store.close()

    def test_main_comparison_uses_ordinary_qa(self):
        from bench.capacity_workflow import FULL_QUEUE_METHODS, full_queue_method_specs
        methods = FULL_QUEUE_METHODS
        self.assertEqual(methods, ("RAG_native", "context_cache_demand", "LlamaIndex_native"))
        specs = full_queue_method_specs(methods, {"gain_asset": "same-reader"},
                                       {"mode": "contextpilot", "checkout": "vendor/contextpilot"})
        baseline = specs[0]["overrides"]["execution"]
        self.assertEqual(baseline["dialogue_mode"], "rag")
        self.assertIsNone(baseline["plan_gain"])
        self.assertEqual(baseline["prefix_retention"], {"mode": "native"})
        self.assertEqual(baseline["cache_compilation"], {"mode": "native"})
