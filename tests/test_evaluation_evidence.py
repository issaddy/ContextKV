"""Scoring uses executed source ranges and the final conclusion provenance."""
from dataclasses import replace
import json

import pytest

from reading_helpers import material

from bench.judge import judge_payload, rule_audit
from bench.quality import answer_evaluation_view, export_evaluation
from controller.candidates import build_plan_candidates
from controller.claim_graph import ClaimGraph
from controller.context_schema import (
    ContextPlan, EvidenceClaim, EvidenceSet, Need, ReadingOutcome, ReadResult, ReadTask, StepRecord, TaskState,
)
from controller.schema import EvidenceSpan, OnlineContext, OnlineTurn, SessionSnapshot
from controller.templates import ByteTemplate


def test_final_excerpt_and_conclusion_only_source_reach_the_scorer(tmp_path):
    first, unit = material("Visible detail. UNSEEN first-block answer.", "first")
    second, second_unit = material("Amount is 7. UNSEEN second-block answer.", "second")
    unread, _ = material("UNSEEN unrelated document.", "unread")
    shown = "Visible detail."
    leaf_text = "Amount is 7."
    unit = replace(unit, text=shown, source_spans=tuple(
        replace(span, end=len(shown)) for span in unit.source_spans))
    leaf = EvidenceClaim("leaf", "n", "extract", text=leaf_text, value=7,
        source_spans=tuple(replace(span, end=len(leaf_text)) for span in second_unit.source_spans))
    root = EvidenceClaim("root", "n", "derived", text="The amount is 7.", value=7,
                         parent_claim_ids=("leaf",), operation="sum")
    state = TaskState("task", "What is the amount?", needs=(Need("n"),), current_turn_id=1)
    graph = ClaimGraph((leaf, root))
    plan = ContextPlan("plan", state, EvidenceSet("selected", (unit,), source_blocks=(first,)),
                       answer_plan=graph.answer_plan(state))
    outcome = ReadingOutcome(plan, claims=(leaf, root))
    snapshot = SessionSnapshot("session", archive=(first, second, unread))
    candidate = build_plan_candidates(snapshot, state.raw_question, plan, ByteTemplate(),
                                     outcome=outcome, actions=("REBUILD",), layouts=("existing",))[0]
    assert candidate.feasible
    assert {span.key for span in candidate.evidence_spans} == {first.key}
    references = {key: ref for ref, key in candidate.reference_map}
    turn = OnlineTurn("session", "task", 1, "fixture", state.raw_question, (),
        tuple(OnlineContext(block.key.passage_id, block.text, block.source_uri, block.title)
              for block in (first, second)), "P3")
    export_evaluation(turn, f"7 [{references[second.key]}]",
        official_path=tmp_path/"official.jsonl", audit_path=tmp_path/"audit.jsonl",
        visible_evidence_manifest=candidate.evidence_spans, audit_evidence=snapshot.archive,
        input_ids=candidate.input_ids, output_ids=(), reading_outcome=outcome,
        reference_map=candidate.reference_map,
        citation_scope=tuple(ref for ref, key in candidate.reference_map if key in candidate.required_blocks))
    official = json.loads((tmp_path/"official.jsonl").read_text())
    audit = json.loads((tmp_path/"audit.jsonl").read_text())
    view = answer_evaluation_view(turn, visible_evidence_manifest=candidate.evidence_spans,
        audit_evidence=snapshot.archive, reading_outcome=outcome,
        reference_map=candidate.reference_map,
        citation_scope=tuple(ref for ref, key in candidate.reference_map if key in candidate.required_blocks))
    assert view == {key: audit[key] for key in view}
    contexts = {row["document_id"]: row["text"] for row in official["contexts"]}
    assert contexts == {first.key.passage_id: shown, second.key.passage_id: leaf_text}
    assert len(audit["audit_evidence"]) == 3
    assert audit["reading_conclusions"]["facts"][0][5] == 7
    assert not rule_audit(audit)["unknown_citations"]
    assert "UNSEEN" not in json.dumps(judge_payload(audit))


def test_source_offsets_and_overlapping_excerpts_are_projected_once(tmp_path):
    block, _ = material("0123456789UNSEEN", "offset")
    block = replace(block, span=(100, 100+len(block.text)))
    spans = tuple(EvidenceSpan(block.key, 1, 0, end-start, block.source_uri, (), "E_ab",
                              source_start=start, source_end=end)
                  for start, end in ((102, 106), (104, 108)))
    turn = OnlineTurn("session", "task", 1, "fixture", "Which digits?", (),
                      (OnlineContext(block.key.passage_id, block.text, block.source_uri),), "P3")
    export_evaluation(turn, "234567 [E_ab]", official_path=tmp_path/"official.jsonl",
        audit_path=tmp_path/"audit.jsonl", visible_evidence_manifest=spans,
        audit_evidence=(block,), input_ids=(), output_ids=(), citation_scope=("E_ab",))
    official = json.loads((tmp_path/"official.jsonl").read_text())
    assert official["contexts"][0]["text"] == "234567"


def test_external_refine_scoring_keeps_only_actual_prior_and_final_source_slices(tmp_path):
    first, _ = material("Earlier fact. UNSEEN early remainder.", "early")
    last, _ = material("Final detail. UNSEEN final remainder.", "last")
    unread, _ = material("UNSEEN retrieved but never supplied.", "unread")
    first = replace(first, span=(100, 100+len(first.text)))
    prior_span = EvidenceSpan(first.key, 1, 0, 13, first.source_uri, (), "E_ab",
                              source_start=100, source_end=113)
    final_span = EvidenceSpan(last.key, 1, 0, 13, last.source_uri, (), "E_cd",
                              source_start=0, source_end=13)
    task = ReadTask("compact:1", "LLAMAINDEX_COMPACT")
    step = StepRecord("attempt", task.step_id, task,
        result=ReadResult(task.step_id, raw_output="Earlier fact [E_ab]."),
        metrics={"source_view": {"input_hash": "actual-prior-input", "evidence_spans": (prior_span,)}})
    state = TaskState("task", "What facts?", current_turn_id=1)
    plan = ContextPlan("external", state, EvidenceSet("sources", (), source_blocks=(first, last, unread)),
                       reading_mode="llamaindex")
    outcome = ReadingOutcome(plan, steps=(step,))
    turn = OnlineTurn("session", "task", 1, "fixture", state.raw_question, (),
        tuple(OnlineContext(block.key.passage_id, block.text, block.source_uri)
              for block in (first, last, unread)), "P3")
    export_evaluation(turn, "Earlier fact [E_ab], final detail [E_cd].",
        official_path=tmp_path/"official.jsonl", audit_path=tmp_path/"audit.jsonl",
        visible_evidence_manifest=(final_span,), audit_evidence=(first, last, unread),
        input_ids=(), output_ids=(), reading_outcome=outcome,
        reference_map=(("E_ab", first.key), ("E_cd", last.key)), citation_scope=("E_ab", "E_cd"))
    official = json.loads((tmp_path/"official.jsonl").read_text())
    audit = json.loads((tmp_path/"audit.jsonl").read_text())
    view = answer_evaluation_view(turn, visible_evidence_manifest=(final_span,),
        audit_evidence=(first, last, unread), reading_outcome=outcome,
        reference_map=(("E_ab", first.key), ("E_cd", last.key)), citation_scope=("E_ab", "E_cd"))
    assert view == {key: audit[key] for key in view}
    assert {row["document_id"]: row["text"] for row in official["contexts"]} == {
        first.key.passage_id: "Earlier fact.", last.key.passage_id: "Final detail."}
    assert audit["conclusion_source_manifest"][0]["execution_proof"] == {
        "step_id": "compact:1", "generation_kind": "LLAMAINDEX_COMPACT", "input_hash": "actual-prior-input"}
    assert audit["reading_conclusions"] == {"kind": "llamaindex_raw_prior_outputs", "outputs": [{
        "step_id": "compact:1", "generation_kind": "LLAMAINDEX_COMPACT",
        "input_hash": "actual-prior-input", "raw_output": "Earlier fact [E_ab]."}]}
    assert not outcome.claims
    assert not rule_audit(audit)["unknown_citations"]
    assert "UNSEEN" not in json.dumps(judge_payload(audit))


@pytest.mark.asyncio
async def test_multidoc_author_f1_and_em_use_existing_offline_overlap(tmp_path):
    from bench.study import write_jsonl
    from bench.study_scoring import score_dataset

    references = [dict(task_id=key, Collection="multidoc2dial", conversation_id="dialogue",
        targets=[{"text": "Another answer."}, {"text": "The red fox."}]) for key in ("one", "two", "absent")]
    write_jsonl(tmp_path/"references.jsonl", references)
    write_jsonl(tmp_path/"official.jsonl", [dict(task_id=key, predictions=[{"text": text}])
        for key, text in (("one", "The red, fox."), ("two", "red"))])
    rows = [dict(task_id=key, status="completed", split="test") for key in ("one", "two")]
    result = await score_dataset({}, {"primary_metric": "f1", "normalization": 1}, rows,
        tmp_path, expected_task_ids=["one", "two", "absent"])
    assert result[0]["metrics"] == {"f1": 1, "exact_match": 1}
    assert result[1]["metrics"] == {"f1": pytest.approx(2/3), "exact_match": 0}
    overlap = json.loads((tmp_path/"scoring/reference-overlap.json").read_text())
    assert overlap["missing_task_ids"] == ["absent"]
    assert overlap["tasks"]["absent"] == {"f1": 0, "exact_match": 0}
    manifest = json.loads((tmp_path/"scoring/manifest.json").read_text())
    assert manifest["judge_cost"]["model_calls"] == 0
    assert manifest["missing_execution_task_ids"] == ["absent"]


@pytest.mark.asyncio
async def test_versionqa_uses_author_binary_accuracy_and_original_answer_fields(tmp_path, monkeypatch):
    from adapters.query_selector import CompletionResult
    from bench.study import write_jsonl
    from bench.study_scoring import score_dataset
    from bench.versionqa import AUTHOR_CORRECTNESS_PROMPT

    requests, configs = [], []

    class Client:
        manifest = {"model": "judge", "revision": "fixed"}

        def __init__(self, config):
            configs.append(config)

        def complete(self, messages):
            requests.append(messages)
            return CompletionResult("completed", "1" if len(requests) == 1 else "0", "stop",
                usage={"prompt_tokens": 20, "completion_tokens": 1}, elapsed_ms=5)

        def close(self):
            pass

    monkeypatch.setattr("adapters.query_selector.OpenAICompletionClient", Client)
    references = [dict(task_id=key, category="Version-Specific", input=[{"text": question}],
        targets=[{"text": target}]) for key, question, target in (
            ("v1", "What changed in v2?", "The limit became 10."),
            ("v2", "Which version introduced it?", "Version 2."))]
    answers = ["The limit became 10 [E_ab].", "Version 1."]
    write_jsonl(tmp_path/"references.jsonl", references)
    write_jsonl(tmp_path/"official.jsonl", [dict(task_id=reference["task_id"], predictions=[{"text": answer}])
        for reference, answer in zip(references, answers)])
    config = {"scoring": {"judge": {"base_url": "http://judge/v1", "model": "judge", "revision": "fixed"}}}
    spec = {"primary_metric": "versionqa_accuracy", "normalization": 1}
    rows = [dict(task_id=reference["task_id"], status="completed", split="test") for reference in references]
    result = await score_dataset(config, spec, rows, tmp_path, expected_task_ids=["v1", "v2"])
    assert [row["metrics"]["versionqa_accuracy"] for row in result] == [1, 0]
    assert requests[0] == [{"role": "system", "content": AUTHOR_CORRECTNESS_PROMPT},
        {"role": "user", "content": "Question: What changed in v2?\nReference Answer: The limit became 10.\n"
                                     "Model Answer: The limit became 10 [E_ab].\nScore:"}]
    assert configs[0].temperature == 0 and configs[0].max_new_tokens == 16
    assert result[0]["scoring"]["identity"]["protocol"] == "versionqa_author_binary_accuracy"
    await score_dataset(config, spec, rows, tmp_path, expected_task_ids=["v1", "v2"])
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_loong_resumes_scores_and_exports_existing_group_summaries(tmp_path, monkeypatch):
    from adapters.query_selector import CompletionResult
    from bench.loong_score import main
    from bench.study import write_jsonl
    from bench.study_scoring import _loong_scores

    requests = []

    class Client:
        manifest = {"model": "judge", "revision": "fixed"}

        def __init__(self, config):
            pass

        def complete(self, messages):
            requests.append(messages)
            return CompletionResult("completed", "Rating: [[50]]. The maximum is [[100]].", "stop",
                usage={"prompt_tokens": 20, "completion_tokens": 2}, elapsed_ms=5)

        def close(self):
            pass

    monkeypatch.setattr("adapters.query_selector.OpenAICompletionClient", Client)
    monkeypatch.setattr("bench.loong_score.OpenAICompletionClient", Client)
    references = [dict(task_id=key, reference_value="Gold", source_task={"type": "financial", "level": 2,
        "set": 1, "question": "Compare.", "instruction": "Report both.",
        "prompt_template": "{docs}\n{question}\n{instruction}"}) for key in ("l1", "l2")]
    write_jsonl(tmp_path/"references.jsonl", references)
    write_jsonl(tmp_path/"official.jsonl", [dict(task_id=key, predictions=[{"text": "Answer"}]) for key in ("l1", "l2")])
    directory = tmp_path/"scoring"
    saved = dict(task_id="l1", domain="financial", level=2, length_set=1, score=100, status="scored")
    write_jsonl(directory/"loong.scores.jsonl", [saved])
    config = {"scoring": {"judge": {"base_url": "http://judge/v1", "model": "judge", "revision": "fixed"}}}
    records, _, outputs, _ = await _loong_scores(config, {}, directory, config["scoring"])
    assert len(requests) == 1 and records[0] == saved
    summary = json.loads((directory/"loong.summary.json").read_text())
    assert summary["summary"]["observed_mean_0_100"] == 75
    assert summary["summary"]["observed_perfect_rate"] == .5
    assert summary["by_domain"]["financial"] == summary["by_level"]["2"] == summary["summary"]
    assert str(directory/"loong.summary.json") in outputs
    await _loong_scores(config, {}, directory, config["scoring"])
    assert len(requests) == 1

    standalone = tmp_path/"standalone"
    write_jsonl(standalone/"scores.jsonl", records)
    judge = tmp_path/"judge.json"
    judge.write_text(json.dumps(config["scoring"]["judge"]))
    report = main(["--references", str(tmp_path/"references.jsonl"), "--predictions", str(tmp_path/"official.jsonl"),
                   "--judge-config", str(judge), "--output", str(standalone)])
    assert report["summary"] == summary["summary"] and len(requests) == 1


def test_saved_reports_keep_dataset_metrics_costs_and_actual_message_roles(tmp_path):
    from bench.plan_result_summary import history_messages, summarize
    from bench.reassess_saved import reassess
    from bench.study import write_json

    methods = ["RAG_native", "context_cache_demand"]
    matrix = {"methods": [{"id": name} for name in methods], "seed": 42,
        "selection": {"bootstrap_samples": 10},
        "datasets": {
            "versionqa": {"primary_metric": "versionqa_accuracy", "normalization": 1},
            "loong": {"primary_metric": "loong_reference_score", "normalization": 100}},
        "cohorts": {"test": {"versionqa": ["v1"], "loong": ["l1"]}}}
    history = [{"turn_id": 1, "user_text": "First question", "assistant_text": None},
        {"turn_id": 2, "user_text": "Clarification", "assistant_text": "First reply"},
        {"turn_id": 3, "user_text": None, "assistant_text": "More detail"}]
    rows = []
    for method in methods:
        directory = tmp_path/method
        write_json(directory/"measurements"/"segment"/"batch_cost.json", {"gpu_gib_seconds": 9})
        for dataset, task_id, metric, score in (("versionqa", "v1", "versionqa_accuracy", 1),
                                               ("loong", "l1", "loong_reference_score", 50)):
            rows.append({"method": method, "dataset": dataset, "task_id": task_id,
                "conversation_id": task_id, "turn": 1, "split": "test", "status": "completed",
                "question": "Question", "answer": "Answer", "actual_history": history,
                "metrics": {metric: score}, "cost": {"complete_ms": 20, "total_input_tokens": 100},
                "generation_dir": str(directory), "summary_sources": [],
                "response": {"attempt_id": method+task_id, "read_count": 0, "reduce_count": 0,
                    "step_metrics": [], "receipt": {"generation_complete": True},
                    "truncated": False, "idempotent_replay": False}})

    report = reassess(matrix, rows, tmp_path/"reassessment", split="test",
        baseline=methods[0], history_protocol="fixed_history")
    assert "track_weights" not in report
    for method in methods:
        result = report["methods"][method]
        assert "queue_quality" not in result
        assert result["datasets"]["versionqa"]["queue_primary_score"] == 1
        assert result["datasets"]["loong"]["queue_primary_score"] == 50
        assert result["datasets"]["loong"]["paid_cost"]["total_input_tokens"]["observed_total"] == 100
        assert result["cost_batches"][0]["batch"] == {"gpu_gib_seconds": 9}
    assert all(pair["fixed_input_pair"] for pair in report["history_pairs"])

    summary, selected, pairs = summarize(matrix,
        {(row["method"], row["dataset"], row["task_id"]): row for row in rows}, methods, "test", None)
    assert "aggregate" not in summary and len(selected) == 4
    assert summary["datasets"]["versionqa"]["methods"][methods[0]]["primary"] == 1
    assert summary["datasets"]["loong"]["methods"][methods[0]]["primary"] == 50
    assert all(pair["same_actual_history"] for pair in pairs)
    assert history_messages(rows[0]) == [("user", "First question"), ("user", "Clarification"),
                                       ("assistant", "First reply"), ("assistant", "More detail")]
