import copy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_model_scope import local_scope as local_scope

import adapters.table_relations as relations
from adapters.mtrag import OnlineContext, OnlineTurn
from adapters.retrieval import to_evidence
from controller.candidates import build_candidates
from controller.evidence import EvidencePolicy, make_evidence
from controller.schema import ExecutionLedger, Message, SessionSnapshot, TurnRecord
from controller.templates import SYSTEM_PROMPT, ByteTemplate, verify_evidence_spans


TABLE = "Table 8. Transfer limits\n\nObject Maximum\n\n Alpha 7 MB\n Beta 9 MB\n"


def committed_snapshot(blocks):
    snapshot = SessionSnapshot("table-session")
    query, answer = "Original question, including every condition.\n", "Full original answer.\nSecond paragraph."
    first = build_candidates(snapshot, query, blocks)[0]
    template = ByteTemplate()
    output = template.tokenizer.encode(answer)
    history = (TurnRecord(1, query, answer, output, evidence_keys=tuple(first.required_blocks)),)
    ledger = ExecutionLedger(1, "native-session", 1,
        first.messages + (Message("assistant", answer),), first.input_ids, output,
        first.visible_blocks, template.fingerprint.template_hash, template.fingerprint.model_revision,
        snapshot.corpus_snapshot_id, first.evidence_spans, first.reference_map)
    return replace(snapshot, version=1, raw_history=history, raw_history_version=1,
                   execution=ledger, archive=blocks, reference_map=first.reference_map)


@pytest.mark.parametrize("layout", ["stable_key", "contextpilot"])
@pytest.mark.parametrize("style", ["all_rows", "focus_refs"])
def test_required_table_annotation_is_shared_once_and_append_reuses_complete_body(monkeypatch, layout, style):
    table = make_evidence(TABLE, "table")
    unrelated = make_evidence(TABLE.replace("Alpha", "Unrelated"), "unrelated")
    current = make_evidence("New original passage.", "current", dependencies=(table.key,))
    snapshot = committed_snapshot((table, unrelated))
    frozen = copy.deepcopy(snapshot)
    parsed = []
    extract = relations.extract_table_relations

    def record(text):
        parsed.append(text)
        return extract(text)

    monkeypatch.setattr(relations, "extract_table_relations", record)
    policy = EvidencePolicy(layout=layout, table_relations=True, table_annotation_style=style)
    append, rebuild = build_candidates(snapshot, "Current exact query?\n", (current,), policy)
    assert sorted(parsed) == sorted([table.text, current.text])  # Shared required set, once per build.
    ref = next(ref for ref, key in append.reference_map if key == table.key)
    if style == "focus_refs":
        from adapters.table_focus import focused_annotation
        annotation, _ = focused_annotation("Current exact query?\n", ((ref, extract(table.text)),))
    else:
        annotation = relations.render_table_relations(ref, extract(table.text))
    assert annotation and all(annotation in c.messages[-1].content for c in (append, rebuild))
    assert all("Unrelated" not in c.messages[-1].content for c in (append, rebuild))
    assert append.required_blocks == rebuild.required_blocks == {table.key, current.key}
    assert append.messages[:-1] == snapshot.execution.messages
    assert table.text not in append.messages[-1].content
    assert append.evidence_spans[:len(snapshot.execution.evidence_spans)] == snapshot.execution.evidence_spans
    assert rebuild.messages[1:3] == (Message("user", snapshot.raw_history[0].user_text),
                                     Message("assistant", snapshot.raw_history[0].assistant_text))
    for candidate in (append, rebuild):
        assert candidate.feasible
        assert candidate.messages[-1].content.endswith("\nQuestion:\nCurrent exact query?\n\n")
        assert candidate.messages[-1].content.index(annotation) < candidate.messages[-1].content.index("\nQuestion:\n")
        visible = verify_evidence_spans(candidate.messages, candidate.evidence_spans,
            (*snapshot.archive, current), candidate.reference_map)
        assert candidate.required_blocks <= visible
        for span in candidate.evidence_spans:
            original = next(b for b in (*snapshot.archive, current) if b.key == span.key)
            assert candidate.messages[span.message_index].content[span.start:span.end] == original.text
    assert snapshot == frozen


def test_repeated_retrieved_table_gets_current_annotation_even_with_no_new_bodies():
    table = make_evidence(TABLE, "table")
    snapshot = committed_snapshot((table,))
    append, rebuild = build_candidates(snapshot, "Repeat the limit?", (table,), EvidencePolicy(table_relations=True))
    assert len(append.evidence_spans) == len(snapshot.execution.evidence_spans)
    assert "<evidence_body>" not in append.messages[-1].content
    assert "Table row associations from [" in append.messages[-1].content
    assert "Table row associations from [" in rebuild.messages[-1].content
    assert table.text in rebuild.messages[-1].content


def test_focus_uses_current_query_without_removing_unselected_required_rows():
    text = "Table 2. Operations\n\nObject Maximum\n\n Width 4 mm\n Duration 7 ms\n"
    block = make_evidence(text, "operations")
    snapshot = committed_snapshot((block,))
    candidates = build_candidates(snapshot, "Not widths, I meant durations?", (block,),
        EvidencePolicy(table_relations=True, table_annotation_style="focus_refs"))
    for candidate in candidates:
        current = candidate.messages[-1].content
        assert 'object="Duration"' in current and 'object="Width"' not in current
        assert candidate.required_blocks == {block.key}
        span = next(s for s in candidate.evidence_spans if s.key == block.key)
        assert candidate.messages[span.message_index].content[span.start:span.end] == text
        assert "Width 4 mm" in candidate.messages[span.message_index].content


@pytest.mark.parametrize("style", ["all_rows", "focus_refs"])
def test_default_off_preserves_shared_message_and_token_output(monkeypatch, style):
    table = make_evidence(TABLE, "table")
    snapshot = SessionSnapshot("s", raw_history=(TurnRecord(1, "Full prior question", "Full prior answer"),))
    def unexpected(_):
        raise AssertionError("disabled feature must not parse evidence")
    monkeypatch.setattr(relations, "extract_table_relations", unexpected)
    implicit = build_candidates(snapshot, "Question?", (table,))[0]
    explicit = build_candidates(snapshot, "Question?", (table,),
        EvidencePolicy(table_relations=False, table_annotation_style=style))[0]
    ref = next(ref for ref, key in implicit.reference_map if key == table.key)
    metadata = json.dumps({"id": ref, "corpus": table.key.corpus_id, "document": table.key.document_id,
        "passage": table.key.passage_id, "version": table.key.version, "source": table.source_uri,
        "title": table.title, "provenance": table.provenance}, ensure_ascii=False, separators=(",", ":"))
    # Shared QA/citation layout, without any optional table annotations.
    from controller.templates import current_source_guide
    guide = current_source_guide("Question?", (table,), implicit.reference_map)
    expected = (Message("system", SYSTEM_PROMPT), Message("user", "Full prior question"),
        Message("assistant", "Full prior answer"), Message("user",
        f"\nEvidence [{ref}] {metadata}\n<evidence_body>\n{TABLE}\n</evidence_body>\n"
        f"\n{guide}\n"
        f"\nCitations allowed for this answer: [{ref}].\n"
        "Use only these sources. Answer supported parts; identify any unsupported part separately.\n"
        "\nQuestion:\nQuestion?\n"))
    rendered = ByteTemplate().render(expected)
    assert implicit.messages == explicit.messages == expected
    assert implicit.input_ids == explicit.input_ids == rendered.input_ids
    assert implicit.input_hash == explicit.input_hash == rendered.input_hash
    assert implicit.evidence_spans == explicit.evidence_spans


@pytest.mark.parametrize("style", ["all_rows", "focus_refs"])
def test_annotation_tokens_use_shared_overflow_without_trimming_qa_or_bodies(style):
    table = make_evidence(TABLE, "table")
    snapshot = committed_snapshot((table,))
    enabled = build_candidates(snapshot, "Current?", (table,),
        EvidencePolicy(table_relations=True, table_annotation_style=style))
    disabled = build_candidates(snapshot, "Current?", (table,))
    output_budget = 32
    limit = min(len(c.input_ids) for c in enabled) + output_budget - 1
    assert max(len(c.input_ids) for c in disabled) + output_budget <= limit
    bounded = build_candidates(snapshot, "Current?", (table,),
        EvidencePolicy(table_relations=True, table_annotation_style=style,
                       max_new_tokens=output_budget, max_total_model_tokens=limit))
    for full, candidate in zip(enabled, bounded):
        assert not candidate.feasible and candidate.reason == "shared_context_policy_overflow"
        assert candidate.messages == full.messages and candidate.input_ids == full.input_ids
        assert candidate.evidence_spans == full.evidence_spans
        assert snapshot.raw_history[0].assistant_text in [m.content for m in candidate.messages]


@pytest.mark.parametrize("style", ["all_rows", "focus_refs"])
def test_table_annotation_reuses_existing_hash_work_and_fingerprints(monkeypatch, style):
    import controller.evidence as evidence
    import controller.templates as templates
    table = make_evidence(TABLE, "table")
    snapshot = committed_snapshot((table,))
    counts = {"content": 0, "identity": 0, "tokens": 0}
    def wrap(kind, fn):
        def counted(value):
            counts[kind] += 1
            return fn(value)
        return counted
    monkeypatch.setattr(evidence, "content_hash", wrap("content", evidence.content_hash))
    monkeypatch.setattr(evidence, "identity_hash", wrap("identity", evidence.identity_hash))
    monkeypatch.setattr(templates, "token_hash", wrap("tokens", templates.token_hash))
    off = build_candidates(snapshot, "Current?", (table,), EvidencePolicy(table_relations=False))
    expected_counts = counts.copy()
    counts.update({k: 0 for k in counts})
    on = build_candidates(snapshot, "Current?", (table,),
        EvidencePolicy(table_relations=True, table_annotation_style=style))
    assert counts == expected_counts == {"content": 1, "identity": 0, "tokens": 2}
    for before, after in zip(off, on):
        assert before.fingerprint == after.fingerprint
        assert before.reference_map == after.reference_map
        assert before.required_blocks == after.required_blocks
        assert before.input_hash != after.input_hash  # Existing final input identity changes normally.


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["all_rows", "focus_refs"])
@pytest.mark.parametrize("structure", ["legacy", "explicit"])
async def test_native_and_controller_share_table_annotated_input_exactly(style, structure):
    from bench.native_baseline import NativeBaseline
    sent = []
    async def generate(ids, **kwargs):
        sent.append(ids)
        return SimpleNamespace(text="answer", output_ids=(1,), finish_reason="stop", truncated=False,
                               meta_info={}, raw_response={}, http_elapsed_ms=1.0, timing=None)
    policy = EvidencePolicy(table_relations=True, table_annotation_style=style, table_structure_mode=structure)
    text = TABLE if structure == "legacy" else TABLE.replace("Table 8. ", "")
    turn = OnlineTurn("s", "task", 1, "test", "Current?", (), (OnlineContext("table", text),), "P3")
    template = ByteTemplate()
    expected = build_candidates(SessionSnapshot("s", corpus_snapshot_id="test-v1"), turn.query,
        tuple(to_evidence(c, "test-v1") for c in turn.contexts), policy, template)[0]
    native = NativeBaseline("B0", SimpleNamespace(stream=False, generate=generate), template,
        policy=policy, corpus_snapshot_id="test-v1", test_mode=True)
    response = await native(turn)
    assert response.input_ids == sent[0] == expected.input_ids
    expected_marker = "Source table facts relevant" if style == "focus_refs" else "Table row associations from ["
    assert expected_marker in bytes(response.input_ids).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("style", ["all_rows", "focus_refs"])
@pytest.mark.parametrize("structure", ["legacy", "explicit"])
async def test_matrix_inherits_shared_table_policy_for_both_paths(tmp_path, monkeypatch, enabled, style, structure, local_scope):
    import bench.experiments as experiments
    import bench.native_baseline as native_module
    import controller.cli as cli
    split, manifest = tmp_path / "split.json", tmp_path / "engine.json"
    split.write_text(json.dumps({"dev": ["session-a"], "test": ["session-b"]}))
    manifest.write_text("{}")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model":{"scope_path":str(local_scope.path)},"context": {"table_relations": enabled, "table_annotation_style": style,
                                                  "table_structure_mode": structure}}))
    spec = {"schema_version": 1, "input": str(Path(__file__).parent / "fixtures/mtrag_small.jsonl"),
        "domain": "all", "split_manifest": str(split), "protocols": ["P2"], "concurrency": [1], "seeds": [42],
        "base_config": str(config_path), "methods": [
            {"id": "native", "kind": "native", "native": {"baseline": "B0", "base_url": "http://fake",
                                                        "engine_manifest": str(manifest)}},
            {"id": "controller"}]}
    seen = {}
    async def native_run(args):
        seen["native"] = (args.table_relations, args.table_annotation_style, args.table_structure_mode)
        (Path(args.output) / "native-fake.json").write_text(json.dumps({"baseline": "B0",
            "engine_manifest_sha256": sha256(Path(args.engine_manifest).read_bytes()).hexdigest()}))
    async def controller_run(args, config):
        seen["controller"] = (config.context.table_relations, config.context.table_annotation_style,
                              config.context.table_structure_mode)
        (Path(config.state_dir) / "benchmark-fake.json").write_text(json.dumps({"engine_kind": "mock"}))
    monkeypatch.setattr(native_module, "_run", native_run)
    monkeypatch.setattr(cli, "benchmark", controller_run)
    monkeypatch.setattr(experiments, "summarize_run", lambda payload: {"measurement_kind": "simulation"})
    monkeypatch.setattr(experiments, "markdown_report", lambda summary: "mock contract")
    for job in experiments.matrix_jobs(spec):
        await experiments.execute_job(spec, job, tmp_path / job["job_id"])
    assert seen == {"native": (enabled, style, structure), "controller": (enabled, style, structure)}


@pytest.mark.parametrize("enabled", [False, True])
def test_native_cli_keeps_explicit_style_separate_from_enable_flag(monkeypatch, enabled):
    import bench.native_baseline as native_module
    from bench.run_sessions import RunReport
    seen = []
    async def fake_run(args):
        policy = native_module.resolve_context_policy(args)
        seen.append((policy.table_relations, policy.table_annotation_style, policy.table_structure_mode))
        return RunReport(mode="closed_loop", seed=42, started_at=0, arrival_window_end=0,
            completed_at=0, turns=(), scheduled_sessions=0, arrived_sessions=0,
            unfinished_turns=0, timed_out_sessions=(), session_makespans=(), time_origin="fixture")
    monkeypatch.setattr(native_module, "_run", fake_run)
    argv = ["unused.jsonl", "--baseline", "B0", "--engine-manifest", "unused.json",
            "--run-config", "unused-scope-config.yaml",
            "--table-annotation-style", "focus_refs", "--table-structure-mode", "explicit"]
    if enabled:
        argv.append("--table-relations")
    assert native_module.main(argv) == 0
    assert seen == [(enabled, "focus_refs", "explicit")]
