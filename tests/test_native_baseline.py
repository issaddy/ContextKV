from __future__ import annotations
import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import unittest

import httpx
import pytest
from test_model_scope import local_scope as local_scope

from adapters.mtrag import OnlineSession, load_rows
from adapters.retrieval import BM25SRetriever, load_beir_corpus, to_evidence
from bench.native_baseline import (NativeBaseline, NativeProtocolError, NativeSGLangClient,
                                   validate_engine_manifest)
from bench.run_sessions import run_sessions
from controller.templates import ByteTemplate
from controller.candidates import CandidatePolicy, build_candidates
from controller.schema import SessionSnapshot

FIXTURE = Path(__file__).parent / "fixtures/mtrag_small.jsonl"


class FakeSSEStream(httpx.AsyncByteStream):
    def __init__(self, frames, *, disconnect=False):
        self.frames, self.disconnect = frames, disconnect

    async def __aiter__(self):
        for frame in self.frames:
            await asyncio.sleep(.002)
            raw = frame if isinstance(frame, str) else json.dumps(frame)
            wire = ("data: " + raw + "\n\n").encode()
            # Exercise line assembly across transport chunks.
            yield wire[:7]
            yield wire[7:]
        if self.disconnect:
            raise httpx.ReadError("injected partial stream disconnect")


class NativeBaselineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.responses = []
        self.finish = "stop"
        self.missing_ids = False
        self.client_http = httpx.AsyncClient(base_url="http://fake.test", transport=httpx.MockTransport(self.respond))
        self.native = NativeSGLangClient("http://fake.test", http_client=self.client_http, audit_prompt_ids=True)
        self.rows = load_rows(FIXTURE)

    async def asyncTearDown(self):
        await self.client_http.aclose()

    async def respond(self, request):
        payload = json.loads(request.content) if request.content else None
        self.requests.append((request.url.path, payload))
        if request.url.path == "/server_info":
            return httpx.Response(200, json={"version": "fixed-source-fake", "enable_session_radix_cache": True})
        if request.url.path == "/close_session":
            return httpx.Response(200)
        if request.url.path != "/generate":
            return httpx.Response(404)
        number = sum(path == "/generate" for path, _ in self.requests)
        text = f"native own answer {number}"
        data = {"text": text, "output_ids": list(text.encode()), "prompt_token_ids": payload["input_ids"],
                "meta_info": {"finish_reason": {"type": self.finish}, "completion_tokens": len(text.encode()),
                              "cached_tokens": 7}}
        if self.missing_ids:
            del data["output_ids"]
        self.responses.append(data)
        return httpx.Response(200, json=data)

    def baseline(self, kind="B0"):
        return NativeBaseline(kind, self.native, ByteTemplate(), corpus_snapshot_id="fixture-v1",
                              test_mode=True, run_id="test-run")

    async def test_custom_system_prompt_matches_shared_candidate_token_for_token(self):
        prompt = "  Audit only the supplied evidence.\nPreserve this trailing space. \n"
        policy = CandidatePolicy(system_prompt=prompt)
        template = ByteTemplate()
        baseline = NativeBaseline("B0",self.native,template,policy=policy,
                                  corpus_snapshot_id="fixture-v1",test_mode=True)
        turn = OnlineSession(self.rows[:3]).current()
        expected = build_candidates(SessionSnapshot(turn.conversation_id,corpus_snapshot_id="fixture-v1"),
            turn.query,tuple(to_evidence(context,"fixture-v1") for context in turn.contexts),policy,template)
        result = await baseline(turn)
        self.assertEqual(result.input_ids,expected[0].input_ids)
        self.assertEqual(tuple(self.requests[-1][1]["input_ids"]),expected[0].input_ids)
        self.assertIn(prompt,bytes(result.input_ids).decode())

    async def test_shared_explicit_revisit_policy_does_not_restore_a_conservative_only_block(self):
        from adapters.mtrag import OnlineContext, OnlineMessage, OnlineTurn
        from bench.native_baseline import resolve_context_policy
        policy = resolve_context_policy(argparse.Namespace(context_policy=CandidatePolicy(
            revisit_mode="explicit", revisit_history_turns=6, reference_prefix_length=6)))
        baseline = NativeBaseline("B0", self.native, ByteTemplate(), policy=policy,
                                  corpus_snapshot_id="fixture-v1", test_mode=True)
        first = OnlineTurn("scope", "scope:1", 1, "fixture", "Tell me the timeout limit", (),
                           (OnlineContext("old", "The timeout limit is 30 seconds.", "docs://old"),), "P2")
        result = await baseline(first)
        second = OnlineTurn("scope", "scope:2", 2, "fixture", "What about that limit?",
            (OnlineMessage("user", first.query), OnlineMessage("assistant", result.assistant_text)),
            (OnlineContext("current", "Unrelated deployment notes.", "docs://current"),), "P2")
        snapshot = baseline.snapshots["scope"]
        evidence = tuple(to_evidence(c, "fixture-v1") for c in second.contexts)
        shared = build_candidates(snapshot, second.query, evidence, policy, ByteTemplate())[0]
        old_default = build_candidates(snapshot, second.query, evidence, CandidatePolicy(), ByteTemplate())[0]
        assert {key.document_id for key in shared.required_blocks} == {"current"}
        assert {key.document_id for key in old_default.required_blocks} == {"old", "current"}
        actual = await baseline(second)
        assert actual.input_ids == shared.input_ids
        assert {block.key.document_id for block in actual.audit_evidence} == {"current"}

    async def test_b0_full_prompt_no_native_or_research_session_fields(self):
        baseline = self.baseline("B0")
        report = await run_sessions(self.rows[:3], baseline, protocol="P2")
        self.assertEqual([r.status for r in report.turns], ["success"] * 3)
        self.assertEqual([path for path, _ in self.requests], ["/generate"] * 3)
        for _, payload in self.requests:
            self.assertNotIn("session_id", payload)
            self.assertNotIn("session_params", payload)
            self.assertFalse(any("contextkv" in name for name in payload))
            self.assertEqual(payload["sampling_params"]["temperature"], 0)
        second = bytes(self.requests[1][1]["input_ids"]).decode()
        self.assertIn("native own answer 1", second)
        self.assertNotIn("REFERENCE ANSWER", second)
        self.assertNotIn("Original passage a.", second)
        self.assertIn("Original passage f.", second)
        snapshot = baseline.snapshots["session-a"]
        self.assertEqual(len(snapshot.raw_history), 3)
        self.assertIsNone(snapshot.execution)
        self.assertEqual(report.turns[0].result.meta_info["cached_tokens"], 7)
        self.assertFalse(hasattr(report.turns[0].result, "receipt"))
        self.assertEqual(await baseline.close_all(), ())

    async def test_b1_stable_native_id_across_rebuilds_and_close_is_dispatch_only(self):
        baseline = self.baseline("B1")
        await baseline.verify_server()
        report = await run_sessions(self.rows, baseline, concurrency=2)
        self.assertEqual(len(report.turns), 6)
        self.assertTrue(all(r.status == "success" for r in report.turns))
        requests = [payload for path, payload in self.requests if path == "/generate"]
        ids = [p["session_id"] for p in requests]
        self.assertEqual(set(ids), {"native:test-run:session-a", "native:test-run:session-b"})
        self.assertEqual(ids.count("native:test-run:session-a"), 3)
        self.assertEqual(ids.count("native:test-run:session-b"), 3)
        close = await baseline.close_all()
        self.assertEqual(len(close), 2)
        self.assertTrue(all(c.acknowledgement == "not_scheduler_ack" and not c.physical_free_observed for c in close))
        self.assertFalse(any("contextkv" in path or path == "/open_session" for path, _ in self.requests))

    async def test_p1_replaces_generated_answer_with_official_history_explicitly(self):
        baseline = self.baseline()
        report = await run_sessions(self.rows[:3], baseline, protocol="P1")
        self.assertTrue(all(r.status == "success" for r in report.turns))
        second = bytes(self.requests[1][1]["input_ids"]).decode()
        self.assertIn("REFERENCE ANSWER 1", second)
        self.assertNotIn("native own answer 1", second)
        self.assertEqual(report.turns[1].result.branch_reason, "reference_history_sync")
        self.assertTrue(all(t.evidence_keys for t in baseline.snapshots["session-a"].raw_history))
        self.assertFalse(baseline.snapshots["session-a"].raw_history[0].assistant_output_ids)

    async def test_explicit_revisit_uses_same_required_policy_and_restores_original(self):
        baseline = self.baseline()
        gate = OnlineSession(self.rows[:3])
        answer = await baseline(gate.current())
        gate.commit(gate.current().task_id, answer.assistant_text)
        reference = next(ref for ref, key in baseline.snapshots["session-a"].reference_map if key.document_id == "a")
        current = replace(gate.current(), query=f"Revisit {reference} and preserve all conditions")
        result = await baseline(current)
        prompt = bytes(result.input_ids).decode()
        self.assertIn("Original passage a. Do not delete conditions.", prompt)
        self.assertEqual(len(result.visible_evidence_manifest), 6)

    async def test_abort_and_missing_actual_ids_never_commit_history(self):
        for finish, missing in ((None, False), ("abort", False), ("stop", True)):
            with self.subTest(finish=finish, missing=missing):
                self.finish, self.missing_ids = finish, missing
                baseline = self.baseline()
                report = await run_sessions(self.rows[:3], baseline)
                self.assertEqual([r.status for r in report.turns], ["failed"])
                self.assertEqual(report.unfinished_turns, 2)
                self.assertEqual(baseline.snapshots, {})
                self.assertTrue(self.native.observations)

    async def test_length_is_completed_but_truncation_is_explicit(self):
        self.finish = "length"
        baseline = self.baseline()
        report = await run_sessions(self.rows[:1], baseline)
        self.assertEqual(report.turns[0].status, "success")
        self.assertTrue(report.turns[0].result.truncated)
        self.assertEqual(report.turns[0].result.finish_reason, "length")

    async def test_p3_real_retrieval_to_native_http_and_generated_history(self):
        from importlib.metadata import PackageNotFoundError, version
        try:
            if version("bm25s") != "0.3.15" or version("numpy") != "2.3.5":
                self.skipTest("pinned retrieval extra not installed")
        except PackageNotFoundError:
            self.skipTest("optional retrieval extra not installed")
        retriever = BM25SRetriever(load_beir_corpus(FIXTURE.with_name("corpus_small.jsonl")), corpus_snapshot_id="fixture-v1")
        baseline = self.baseline("B1")
        report = await run_sessions(self.rows, baseline, protocol="P3", concurrency=2, retriever=retriever)
        self.assertEqual([r.status for r in report.turns], ["success"] * 6)
        self.assertTrue(all([c.document_id for c in r.online_turn.contexts] == list("abcde") for r in report.turns))
        for _, payload in self.requests:
            if payload and "input_ids" in payload:
                prompt = bytes(payload["input_ids"]).decode()
                self.assertIn("common identical words", prompt)
                self.assertNotIn("Original passage", prompt)
                self.assertNotIn("REFERENCE ANSWER", prompt)

    async def test_formal_mode_requires_engine_manifest_and_real_template(self):
        self.assertTrue(validate_engine_manifest({}, "B0", ByteTemplate()))
        with self.assertRaisesRegex(ValueError, "explicit Qwen3"):
            NativeBaseline("B0", self.native, ByteTemplate(), corpus_snapshot_id="fixture-v1")

    async def test_unmeasured_example_manifests_are_rejected(self):
        for baseline in ("B0", "B1"):
            manifest = {"example_only":True}
            errors = validate_engine_manifest(manifest, baseline)
            self.assertIn("example manifest cannot authorize a formal hardware run", errors)
            self.assertTrue(any("gpu_memory_bytes" in error for error in errors))
            self.assertTrue(any("model.name" in error for error in errors))
            manifest["example_only"] = False
            self.assertTrue(validate_engine_manifest(manifest, baseline))

    async def test_history_mismatch_is_rejected(self):
        baseline = self.baseline()
        gate = OnlineSession(self.rows[:3])
        result = await baseline(gate.current())
        gate.commit(gate.current().task_id, "different invented history")
        self.assertTrue(result.assistant_text)
        with self.assertRaisesRegex(NativeProtocolError, "history differs"):
            await baseline(gate.current())
        self.assertEqual(len(baseline.snapshots["session-a"].raw_history), 1)

    async def test_native_sse_observes_new_token_chunks_and_commits_only_after_done(self):
        received = []
        def respond(request):
            received.append(json.loads(request.content))
            partial = {"text": "A", "output_ids": [65], "meta_info": {"finish_reason": None}}
            two = {"text": "AB", "output_ids": [65, 66], "meta_info": {"finish_reason": None}}
            terminal = {**two, "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": 2}}
            return httpx.Response(200, stream=FakeSSEStream([partial, two, two, terminal, "[DONE]"]))
        async with httpx.AsyncClient(base_url="http://fake.test", transport=httpx.MockTransport(respond)) as http:
            client = NativeSGLangClient("http://fake.test", http_client=http, stream=True)
            baseline = NativeBaseline("B0", client, ByteTemplate(), corpus_snapshot_id="fixture-v1", test_mode=True)
            result = await baseline(OnlineSession(self.rows[:3]).current())
        self.assertTrue(received[0]["stream"])
        self.assertEqual(result.assistant_text, "AB")
        self.assertEqual(result.output_ids, (65, 66))
        self.assertEqual(result.timing.token_observation, "chunk")
        self.assertLess(result.timing.t_submit, result.timing.t_first)
        self.assertLess(result.timing.t_first, result.timing.t_last_token)
        self.assertLess(result.timing.t_last_token, result.timing.t_response_done)
        self.assertLessEqual(result.timing.t_response_done, result.timing.t_commit)
        self.assertEqual(len(baseline.snapshots["session-a"].raw_history), 1)
        self.assertEqual(client.observations[0]["stream_frames"], 4)

    async def test_partial_terminal_or_transport_interruption_cannot_commit(self):
        partial = {"text": "A", "output_ids": [65], "meta_info": {"finish_reason": None}}
        final = {**partial, "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": 1}}
        for frames, disconnect in (([partial, "[DONE]"], False), ([final], False), ([partial], True)):
            with self.subTest(frames=frames, disconnect=disconnect):
                def respond(request):
                    return httpx.Response(200, stream=FakeSSEStream(frames, disconnect=disconnect))
                async with httpx.AsyncClient(base_url="http://fake.test", transport=httpx.MockTransport(respond)) as http:
                    client = NativeSGLangClient("http://fake.test", http_client=http, stream=True)
                    baseline = NativeBaseline("B0", client, ByteTemplate(), corpus_snapshot_id="fixture-v1", test_mode=True)
                    report = await run_sessions(self.rows[:3], baseline)
                    self.assertEqual([r.status for r in report.turns], ["failed"])
                    self.assertEqual(report.unfinished_turns, 2)
                    self.assertEqual(baseline.snapshots, {})
                    self.assertEqual(client.observations[0]["partial_output_tokens"], 1)

    async def test_native_cumulative_contract_rejects_incremental_server(self):
        def respond(request):
            return httpx.Response(200, json={"incremental_streaming_output": True})
        async with httpx.AsyncClient(base_url="http://fake.test", transport=httpx.MockTransport(respond)) as http:
            client = NativeSGLangClient("http://fake.test", http_client=http, stream=True)
            with self.assertRaisesRegex(NativeProtocolError, "incremental_streaming_output=false"):
                await client.server_info()


@pytest.mark.parametrize("failed_generation", [False, True])
def test_native_cli_prompt_file_preserves_body_in_policy_and_workload(tmp_path,monkeypatch,failed_generation,local_scope):
    import bench.native_baseline as native_module
    raw = b"  Use only the cited evidence.\r\nKeep this space. \r\n"
    prompt_file = tmp_path/"prompt.txt"
    prompt_file.write_bytes(raw)
    manifest = tmp_path/"engine.json"
    manifest.write_text("{}")
    context_config = tmp_path/"context.yaml"
    context_config.write_text(f"model:\n  scope_path: {local_scope.path}\ncontext:\n  revisit_mode: explicit\n  revisit_history_turns: 6\n"
                              "  allow_version_comparison: true\n  reference_prefix_length: 6\n"
                              "  max_new_tokens: 123\n  table_relations: true\n"
                              "  table_annotation_style: focus_refs\n")
    observed = []
    def respond(request):
        if request.url.path == "/server_info":
            return httpx.Response(200,json={"version":"fake-http-correctness"})
        payload = json.loads(request.content)
        observed.append(payload)
        response = {"text":"ok", "meta_info":{"finish_reason":{"type":"stop"},"completion_tokens":2}}
        if not failed_generation:
            response["output_ids"] = [111,107]
        return httpx.Response(200,json=response)
    http = httpx.AsyncClient(base_url="http://fake.test",transport=httpx.MockTransport(respond))
    client_class,baseline_class = native_module.NativeSGLangClient,native_module.NativeBaseline
    monkeypatch.setattr("controller.config.create_model_template",lambda *args,**kwargs:ByteTemplate())
    monkeypatch.setattr(native_module,"NativeSGLangClient",lambda *args,**kwargs:
                        client_class("http://fake.test",http_client=http,stream=False))
    monkeypatch.setattr(native_module,"NativeBaseline",lambda *args,**kwargs:
                        baseline_class(*args,**kwargs,test_mode=True,run_id="prompt-audit"))
    output = tmp_path/"native"
    try:
        assert native_module.main([str(FIXTURE),"--baseline","B0","--engine-manifest",str(manifest),
            "--domain","all","--limit-sessions","1","--output",str(output),
            "--system-prompt-file",str(prompt_file), "--run-config",str(context_config)]) == int(failed_generation)
    finally:
        asyncio.run(http.aclose())
    payload = json.loads((output/"native-B0-prompt-audit.json").read_text())
    assert payload["policy"]["system_prompt"] == raw.decode()
    assert payload["workload"]["system_prompt"] == raw.decode()
    assert payload["workload"]["system_prompt_file"] == str(prompt_file)
    assert payload["policy"]["revisit_mode"] == "explicit"
    assert payload["policy"]["revisit_history_turns"] == payload["policy"]["reference_prefix_length"] == 6
    assert payload["policy"]["allow_version_comparison"] is True
    assert payload["policy"]["max_new_tokens"] == 123
    assert payload["policy"]["table_relations"] is True
    assert payload["policy"]["table_annotation_style"] == "focus_refs"
    if failed_generation:
        assert payload["report"]["turns"][0]["status"] == "failed"
        assert payload["report"]["unfinished_turns"] == 2
    assert raw.decode() in bytes(observed[0]["input_ids"]).decode()
    with pytest.raises(SystemExit) as exc:
        native_module.main([str(FIXTURE),"--baseline","B0","--engine-manifest",str(manifest),
            "--system-prompt","one","--system-prompt-file",str(prompt_file)])
    assert exc.value.code == 2


def test_controller_cli_failure_is_nonzero_and_keeps_report(tmp_path):
    from controller.cli import main
    config = tmp_path/"small.yaml"
    config.write_text("context:\n  max_total_model_tokens: 200\n  max_new_tokens: 64\n")
    output = tmp_path/"failed-controller"
    with pytest.raises(SystemExit) as caught:
        main(["--config",str(config), "--state-dir",str(output), "bench",str(FIXTURE),
              "--domain","all", "--limit-sessions","1"])
    assert caught.value.code == 1
    report = json.loads(next(output.glob("benchmark-*.json")).read_text())["report"]
    assert report["turns"][0]["status"] == "failed" and report["unfinished_turns"] == 2
    assert next(output.glob("summary-*.json")).exists()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("configured", [False, True])
def test_native_optional_sampling_preserves_default_wire_and_audits_actual_recipe(stream, configured):
    async def scenario():
        received = []
        recipe = {"temperature": .7, "top_p": .8, "top_k": 20, "min_p": 0}
        frozen_recipe = dict(recipe)

        def respond(request):
            received.append(json.loads(request.content))
            terminal = {"text": "A", "output_ids": [65], "prompt_token_ids": [12, 13],
                        "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": 1}}
            if stream:
                partial = {"text": "A", "output_ids": [65], "meta_info": {"finish_reason": None}}
                return httpx.Response(200, stream=FakeSSEStream([partial, terminal, "[DONE]"]))
            return httpx.Response(200, json=terminal)

        async with httpx.AsyncClient(base_url="http://fake.test", transport=httpx.MockTransport(respond)) as http:
            client = NativeSGLangClient("http://fake.test", http_client=http, stream=stream,
                                       audit_prompt_ids=True, sampling_params=recipe if configured else None)
            # The caller's mutable configuration must not change a running client.
            recipe["temperature"] = .2
            recipe["max_new_tokens"] = 999
            for cap in (1, 3):
                result = await client.generate([12, 13], max_new_tokens=cap, rid=f"case:{cap}")
                assert result.output_ids == (65,)
                assert result.finish_reason == "stop" and not result.truncated
                assert received[-1] == {
                    "input_ids": [12, 13], "stream": stream, "return_prompt_token_ids": True,
                    "sampling_params": {**(frozen_recipe if configured else {"temperature": 0}),
                                        "max_new_tokens": cap}, "rid": f"case:{cap}"}
                assert client.observations[-1]["request"] == received[-1]
            assert result.timing.token_observation == ("chunk" if stream else "none")

    asyncio.run(scenario())


@pytest.mark.parametrize("settings", [
    {"max_new_tokens": 2048}, {"sampling_seed": 42}, {"ignore_eos": True},
    {"temperature": float("nan")}, {"temperature": -1}, {"temperature": True},
    {"top_p": 0}, {"top_p": 1.1}, {"min_p": float("inf")}, {"min_p": -0.1},
    {"top_k": 0}, {"top_k": 20.0},
])
def test_native_sampling_settings_rejected_at_constructor_before_http(settings, monkeypatch):
    def unexpected_http_client(*args, **kwargs):
        raise AssertionError("invalid configuration must fail before opening HTTP client")
    monkeypatch.setattr(httpx, "AsyncClient", unexpected_http_client)
    with pytest.raises(ValueError):
        NativeSGLangClient("http://fake.test", sampling_params=settings)


@pytest.mark.parametrize("stream", [False, True])
def test_native_configured_sampling_still_rejects_output_exceeding_call_budget(stream):
    async def scenario():
        def respond(request):
            assert json.loads(request.content)["sampling_params"]["max_new_tokens"] == 1
            terminal = {"text": "AB", "output_ids": [65, 66],
                        "meta_info": {"finish_reason": {"type": "length"}, "completion_tokens": 2}}
            return (httpx.Response(200, stream=FakeSSEStream([terminal, "[DONE]"])) if stream
                    else httpx.Response(200, json=terminal))
        async with httpx.AsyncClient(base_url="http://fake.test", transport=httpx.MockTransport(respond)) as http:
            client = NativeSGLangClient("http://fake.test", http_client=http, stream=stream,
                sampling_params={"temperature": .7, "top_p": .8, "top_k": 20, "min_p": 0})
            with pytest.raises(NativeProtocolError, match="shared generation budget"):
                await client.generate([12, 13], max_new_tokens=1)
            assert client.observations[-1]["request"]["sampling_params"]["max_new_tokens"] == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_native_json_schema_is_per_call_and_keeps_actual_prompt_audit(stream):
    async def scenario():
        received = []
        schema = {"type": "object", "properties": {"unit": {"type": "integer", "enum": [0, 1]}},
                  "required": ["unit"], "additionalProperties": False}
        original = json.loads(json.dumps(schema))
        def respond(request):
            received.append(json.loads(request.content))
            terminal = {"text": '{"unit":0}', "output_ids": [11], "prompt_token_ids": [12, 13],
                        "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": 1}}
            return (httpx.Response(200, stream=FakeSSEStream([terminal, "[DONE]"])) if stream
                    else httpx.Response(200, json=terminal))
        async with httpx.AsyncClient(base_url="http://fake.test", transport=httpx.MockTransport(respond)) as http:
            client = NativeSGLangClient("http://fake.test", http_client=http, stream=stream,
                                       audit_prompt_ids=True, sampling_params={"temperature": .7})
            await client.generate([12, 13], max_new_tokens=17, json_schema=schema)
            assert schema == original
            assert json.loads(received[-1]["sampling_params"]["json_schema"]) == original
            assert received[-1]["sampling_params"]["max_new_tokens"] == 17
            schema["properties"]["unit"]["enum"].append(9)
            assert json.loads(client.observations[0]["request"]["sampling_params"]["json_schema"]) == original
            await client.generate([12, 13], max_new_tokens=8)
            assert received[-1]["sampling_params"] == {"temperature": .7, "max_new_tokens": 8}
            assert all(r["input_ids"] == [12, 13] and r["return_prompt_token_ids"] for r in received)
    asyncio.run(scenario())


def test_native_grammar_error_is_retained_without_retry_or_unconstrained_fallback():
    async def scenario():
        calls = []
        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(400, json={"error": "grammar compilation failed"})
        async with httpx.AsyncClient(base_url="http://fake.test", transport=httpx.MockTransport(respond)) as http:
            client = NativeSGLangClient("http://fake.test", http_client=http)
            with pytest.raises(httpx.HTTPStatusError):
                await client.generate([12], max_new_tokens=8, json_schema={"type": "unknown_type"})
            assert len(calls) == 1
            assert client.observations[-1]["http_status"] == 400
            assert json.loads(calls[0]["sampling_params"]["json_schema"]) == {"type": "unknown_type"}
    asyncio.run(scenario())
