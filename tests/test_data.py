from __future__ import annotations
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import threading
import unittest

from adapters.mtrag import (OnlineMessage, OnlineSession, json_hash, load_rows,
                            online_turn, profile_rows, stratified_split)
from adapters.retrieval import (BM25SRetriever, fixed_retrieve, load_beir_corpus,
                                retrieval_query, to_evidence)
from bench.run_sessions import run_sessions

FIXTURE = Path(__file__).parent / "fixtures" / "mtrag_small.jsonl"


class DataTests(unittest.TestCase):
    def setUp(self):
        self.rows = load_rows(FIXTURE)

    def test_whitelist_protocols_and_no_reference_markers(self):
        row = self.rows[1]
        p1 = online_turn(row, "P1")
        self.assertEqual(p1.history[-1].text, "REFERENCE ANSWER 1")
        self.assertEqual(p1.branch_reason, "reference_history_sync")
        history = (OnlineMessage("user", "actual question"), OnlineMessage("assistant", "own answer"))
        p2 = online_turn(row, "P2", history)
        rendered = json.dumps(asdict(p2))
        self.assertNotIn("REFERENCE", rendered)
        self.assertNotIn("GOLD", rendered)
        self.assertNotIn("LABEL", rendered)
        self.assertNotIn("reference", rendered)
        self.assertEqual(p2.history, history)
        self.assertEqual(online_turn(row, "P3", history).contexts, ())
        self.assertEqual(p2.task_id, "session-a<::>2")

    def test_sequential_gate_never_reveals_future(self):
        gate = OnlineSession(self.rows[:3])
        self.assertNotIn("question 2", json.dumps(asdict(gate.current())))
        with self.assertRaises(ValueError):
            gate.commit("session-a<::>2", "bad")
        gate.commit(gate.current().task_id, "own answer")
        self.assertEqual(gate.current().turn, 2)
        self.assertEqual(gate.current().history[-1].text, "own answer")
        self.assertEqual(gate.current(), gate.current())

    def test_profile_uses_content_hash_and_session_split(self):
        result = profile_rows(self.rows)
        self.assertEqual(result["tasks"], 6)
        self.assertEqual(result["sessions"], 2)
        self.assertEqual(result["current_blocks_after_first"], 20)
        self.assertEqual(result["adjacent_repeat"], 10)
        self.assertEqual(result["ever_repeat"], 12)
        manifest = stratified_split(self.rows)
        self.assertEqual(manifest, stratified_split(list(reversed(self.rows))))
        self.assertFalse(set(manifest["dev"]) & set(manifest["test"]))
        self.assertEqual(set(manifest["dev"]) | set(manifest["test"]), {"session-a", "session-b"})
        digest = manifest.pop("manifest_sha256")
        self.assertEqual(digest, json_hash(manifest))

    def test_evidence_preserves_text_identity_and_provenance(self):
        context = online_turn(self.rows[0]).contexts[0]
        block = to_evidence(context, "fixed-snapshot")
        self.assertEqual(block.text, context.text)
        self.assertEqual(block.key.content_sha256, context.content_sha256)
        self.assertEqual(block.key.version, "fixed-snapshot")
        self.assertTrue(block.provenance)
        self.assertEqual(len(fixed_retrieve(online_turn(self.rows[0]))), 5)
        with self.assertRaises(ValueError):
            fixed_retrieve(online_turn(self.rows[0], "P3"))
        self.assertEqual(retrieval_query(["a", "b", "c", "d"]), "b\nc\nd")

    def test_runner_is_serial_per_session_concurrent_between_sessions(self):
        async def check():
            active = set()
            maximum = 0
            calls = []
            async def handler(turn):
                nonlocal maximum
                self.assertNotIn(turn.conversation_id, active)
                active.add(turn.conversation_id)
                maximum = max(maximum, len(active))
                self.assertEqual(len(turn.history), (turn.turn - 1) * 2)
                if turn.turn > 1:
                    self.assertEqual(turn.history[-1].text, f"own {turn.turn-1}")
                await asyncio.sleep(.001)
                calls.append(turn.task_id)
                active.remove(turn.conversation_id)
                return {"assistant_text": f"own {turn.turn}"}
            report = await run_sessions(self.rows, handler, concurrency=2)
            self.assertEqual(len(report.turns), 6)
            self.assertEqual(report.unfinished_turns, 0)
            self.assertEqual(maximum, 2)
            self.assertEqual(report.time_origin, "fixed_retrieval_input_ready")
        asyncio.run(check())

    def test_failed_turn_stops_session_and_counts_dependent_work(self):
        async def check():
            async def fail(turn):
                raise RuntimeError("engine failure")
            report = await run_sessions(self.rows[:3], fail)
            self.assertEqual([r.status for r in report.turns], ["failed"])
            self.assertEqual(report.unfinished_turns, 2)
        asyncio.run(check())

    def test_session_arrivals_and_timeout_are_not_silently_dropped(self):
        async def check():
            async def slow(turn):
                await asyncio.sleep(1)
                return "answer"
            report = await run_sessions(self.rows, slow, concurrency=2, arrival_rate=100000,
                                        arrival_window=.001, drain_timeout=.01)
            self.assertEqual(report.mode, "session_open_loop")
            self.assertEqual(report.arrived_sessions, 2)
            self.assertEqual(len(report.timed_out_sessions), 2)
            self.assertEqual(len(report.turns) + report.unfinished_turns, 6)
        asyncio.run(check())

    def test_timeout_during_think_time_does_not_duplicate_committed_turn(self):
        async def check():
            async def handler(turn):
                return "committed answer"
            report = await run_sessions(self.rows[:3], handler, think_time=1, drain_timeout=.01)
            self.assertEqual([r.status for r in report.turns], ["success"])
            self.assertEqual(report.unfinished_turns, 2)
            self.assertEqual(report.timed_out_sessions, ("session-a",))
        asyncio.run(check())

    def test_p3_drain_cancellation_waits_for_retrieval_before_owned_close(self):
        for use_provider in (False, True):
            with self.subTest(use_provider=use_provider):
                async def check():
                    started, release = threading.Event(), threading.Event()
                    actions, generated, provider_threads = [], [], []
                    context = object()
                    main_thread = threading.get_ident()

                    class Retriever:
                        closed = False

                        def retrieve_turn(inner, turn, *, prior_context=None):
                            self.assertIs(prior_context, context if use_provider else None)
                            started.set()
                            self.assertTrue(release.wait(5), "test did not release retrieval")
                            self.assertFalse(inner.closed, "retrieval used a closed model")
                            actions.append("retrieval_finished")
                            return turn

                        def close(inner):
                            inner.closed = True
                            actions.append("closed")

                    retriever = Retriever()

                    def provider(turn):
                        provider_threads.append(threading.get_ident())
                        self.assertEqual(turn.history, ())
                        return context

                    async def handler(turn):
                        generated.append(turn.task_id)
                        return "must not be committed"

                    async def owner():
                        try:
                            return await run_sessions(self.rows[:3], handler, protocol="P3",
                                retriever=retriever, drain_timeout=.01,
                                retrieval_context_provider=provider if use_provider else None)
                        finally:
                            retriever.close()

                    task = asyncio.create_task(owner())
                    try:
                        self.assertTrue(await asyncio.to_thread(started.wait, 5))
                        await asyncio.sleep(.04)  # Let the declared drain deadline cancel this turn.
                        self.assertFalse(task.done())
                        self.assertFalse(retriever.closed)
                        self.assertEqual(generated, [])
                    finally:
                        release.set()
                        report = await task
                    self.assertEqual(actions, ["retrieval_finished", "closed"])
                    self.assertEqual(generated, [])
                    self.assertEqual([r.status for r in report.turns], ["cancelled"])
                    self.assertEqual(report.turns[0].online_turn.history, ())
                    self.assertIsNone(report.turns[0].assistant_text)
                    self.assertEqual(report.unfinished_turns, 2)
                    self.assertEqual(report.timed_out_sessions, ("session-a",))
                    self.assertEqual(len(provider_threads), int(use_provider))
                    self.assertTrue(all(thread != main_thread for thread in provider_threads))

                asyncio.run(check())

    def test_sink_failure_does_not_reclassify_committed_generation(self):
        async def check():
            observed = []
            async def handler(turn):
                return "committed answer"
            def broken_sink(result):
                observed.append(result.status)
                raise RuntimeError("audit disk failed")
            with self.assertRaisesRegex(RuntimeError, "audit disk"):
                await run_sessions(self.rows[:3], handler, on_result=broken_sink)
            self.assertEqual(observed, ["success"])
        asyncio.run(check())

    def test_p3_retrieval_generation_and_durable_commit_pipeline(self):
        from importlib.metadata import PackageNotFoundError, version
        try:
            if version("bm25s") != "0.3.15" or version("numpy") != "2.3.5":
                self.skipTest("pinned retrieval extra not installed")
        except PackageNotFoundError:
            self.skipTest("optional retrieval extra not installed")
        import tempfile
        from controller.config import RunConfig, create_controller

        async def check(directory):
            service = create_controller(RunConfig(state_dir=directory))
            retriever = BM25SRetriever(load_beir_corpus(FIXTURE.with_name("corpus_small.jsonl")),
                                      corpus_snapshot_id="fixture-corpus-v1")
            async def handler(turn):
                self.assertEqual([c.document_id for c in turn.contexts], list("abcde"))
                self.assertTrue(all(c.text == "common identical words" for c in turn.contexts))
                self.assertNotIn("REFERENCE", json.dumps(asdict(turn)))
                response = await service.serve_turn(turn.conversation_id, turn.query,
                            tuple(to_evidence(c, "fixture-corpus-v1") for c in turn.contexts),
                            request_id=turn.task_id, corpus_snapshot_id="fixture-corpus-v1")
                snapshot = service.store.read_snapshot(turn.conversation_id)
                self.assertEqual(len(snapshot.raw_history), turn.turn)
                self.assertEqual(snapshot.raw_history[-1].assistant_text, response.assistant_text)
                return response
            try:
                report = await run_sessions(self.rows, handler, protocol="P3", concurrency=2,
                                            retriever=retriever)
                self.assertEqual([r.status for r in report.turns], ["success"] * 6)
                self.assertEqual(report.unfinished_turns, 0)
                self.assertEqual(report.time_origin, "pipeline_arrival")
                for result in report.turns:
                    self.assertIsNotNone(result.t_retrieval_done)
                    self.assertGreaterEqual(result.t_retrieval_done, result.t_arrive)
                    if result.turn > 1:
                        previous = next(r for r in report.turns if r.conversation_id == result.conversation_id and r.turn == result.turn - 1)
                        self.assertEqual(result.online_turn.history[-1].text, previous.assistant_text)
                for sid in ("session-a", "session-b"):
                    await service.close_session(sid)
            finally:
                await service.aclose()
                service.store.close()
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(check(directory))

    def test_bm25s_ties_and_original_text_when_extra_is_installed(self):
        from importlib.metadata import PackageNotFoundError, version
        try:
            if version("bm25s") != "0.3.15" or version("numpy") != "2.3.5":
                self.skipTest("pinned retrieval extra not installed")
        except PackageNotFoundError:
            self.skipTest("optional retrieval extra not installed")
        corpus = load_beir_corpus(FIXTURE.with_name("corpus_small.jsonl"))
        retriever = BM25SRetriever(corpus, corpus_snapshot_id="test")
        rows = retriever.retrieve(["common"])
        self.assertEqual([r.document_id for r in rows], list("abcde"))
        self.assertTrue(all(r.text == "common identical words" for r in rows))
        self.assertEqual(retriever.retrieve(["the and"]), ())

    def test_saved_index_load_reuses_index_and_partial_topk_matches_full_sort(self):
        import tempfile
        from unittest.mock import patch
        try:
            import numpy as np
            import bm25s
        except ImportError:
            self.skipTest("optional retrieval extra not installed")
        from adapters.retrieval import deterministic_top_k
        rng = np.random.default_rng(42)
        for size in (1, 4, 5, 100, 10000):
            # Quantized values force many ties, including the kth boundary.
            scores = rng.integers(0, 5, size=size).astype(float)
            expected = np.lexsort((np.arange(size), -scores))[:5]
            self.assertEqual(deterministic_top_k(scores, 5, np).tolist(), expected.tolist())
        retriever = BM25SRetriever(load_beir_corpus(FIXTURE.with_name("corpus_small.jsonl")), corpus_snapshot_id="saved")
        expected = retriever.retrieve(["common"])
        with tempfile.TemporaryDirectory() as directory:
            retriever.save(directory)
            with patch.object(bm25s.BM25, "index", side_effect=AssertionError("must not rebuild")):
                loaded = BM25SRetriever.load(directory)
            self.assertEqual(loaded.retrieve(["common"]), expected)


if __name__ == "__main__":
    unittest.main()


def test_selected_turn_reference_benchmark_keeps_variable_original_contexts():
    from copy import deepcopy
    from bench.quality import official_record
    row = deepcopy(load_rows(FIXTURE)[1])
    for count in (0, 2, 12):
        row['contexts'] = [{'document_id':f'd{i}','text':f'Original {i}'} for i in range(count)]
        gate = OnlineSession([row], 'P1')
        turn = gate.current()
        assert turn.history[-1].text == 'REFERENCE ANSWER 1'
        assert [c.text for c in turn.contexts] == [c['text'] for c in row['contexts']]
        assert len(official_record(turn, 'Protocol fixture')['contexts']) == count
        gate.commit(turn.task_id, 'Protocol fixture')
        assert gate.done
    for protocol in ('P2', 'P3'):
        with unittest.TestCase().assertRaisesRegex(ValueError, 'selected-turn benchmarks use P1'):
            OnlineSession([row], protocol)
