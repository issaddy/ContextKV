"""Scoring uses executed source ranges and the final conclusion provenance."""
from dataclasses import replace
import json

from reading_helpers import material

from bench.judge import judge_payload, rule_audit
from bench.quality import export_evaluation
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
