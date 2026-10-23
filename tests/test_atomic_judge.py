from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bench.atomic_judge import AtomicJudgeClient, evaluate_file
from bench.native_judge import NativeJudgeConfig
from controller.templates import token_hash


def config(**kwargs):
    return NativeJudgeConfig(**({"provider":"test", "model":"selene", "revision":"fixed-revision",
        "base_url":"http://test/v1", "base_model":"llama-base", "base_family":"llama"} | kwargs))


def audit(answer="The cache limit is 8 MiB [E_01].", sources=None):
    sources = sources or ["The cache limit is 8 MiB.", "The timeout is 30 seconds."]
    return {"task_id":"s::1", "conversation_id":"s", "query":"What is the cache limit?",
        "answer":answer, "actual_history":[{"speaker":"assistant", "text":"Earlier I claimed 99 MiB."}],
        "retrieved_evidence":[],
        "visible_evidence_manifest":[{"reference_id":f"E_{i+1:02x}", "key":{"document_id":str(i)}} for i in range(len(sources))],
        "audit_evidence":[{"key":{"document_id":str(i)}, "text":text} for i,text in enumerate(sources)],
        "input_ids":[3,4], "measurement_kind":"simulation", "truncated":False,
        "targets":[{"text":"GOLD_MUST_NOT_APPEAR"}], "source_arm_for_offline_review":"HIDDEN_ARM"}


def extraction(answer, **updates):
    claim={"claim_id":"c1", "unit_id":"u1", "answer_quote":answer, "statement":answer,
           "kind":"assertion", "citation_ids":[], "object":"cache", "relation":"limit",
           "value_raw":"8", "unit_raw":"MiB", "comparator":None, "scope":None,
           "conditions":[], "negation":False, "modality":"asserted"}
    claim.update(updates)
    return {"claims":[claim], "uncovered_units":[]}


def checked(verdict="entailed", *, reference_id="E_01", quote="The cache limit is 8 MiB.",
            relation="supports", faithful=True, **kwargs):
    return {"claim_id":"c1", "extraction_faithful":faithful, "verdict":verdict,
        "evidence":[{"reference_id":reference_id, "quote":quote, "relation":relation}] if quote else [],
        "failed_slots":[], "reason":"Fake transport contract only.", "source_answers_current_question":None} | kwargs


class Responses:
    def __init__(self, *values):
        self.values=iter(values)
        self.requests=[]

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        value=next(self.values)
        if isinstance(value,httpx.Response):
            return value
        return httpx.Response(200,json={"id":"fake", "model":"selene", "choices":[{
            "message":{"content":json.dumps(value)}, "finish_reason":"stop"}],
            "usage":{"prompt_tokens":20,"completion_tokens":10}})


async def run(record, *responses, **kwargs):
    transport=Responses(*responses)
    client=AtomicJudgeClient(config(),transport=httpx.MockTransport(transport),**kwargs)
    try:
        result=await client.evaluate(record)
    finally:
        await client.aclose()
    return result,transport.requests


@pytest.mark.asyncio
async def test_pipeline_binds_answer_and_sources_without_leaking_offline_labels():
    record=audit()
    result,requests=await run(record,extraction(record["answer"]),checked(),checked())
    assert result["status"]=="complete" and result["diagnostic_only"] is True
    assert result["source"]=="atomic_judge_diagnostic" and "scores" not in result
    assert result["formal_quality_eligible"] is False and result["summary"]["quality_score"] is None
    assert result["input_hash"]==token_hash([3,4]) and result["generation_measurement_kind"]=="simulation"
    assert len(requests)==3
    inputs=[json.loads(r["messages"][1]["content"]) for r in requests]
    assert "sources" not in inputs[0] and "actual_history" in inputs[0]
    assert len(inputs[1]["sources"])==2 and len(inputs[2]["sources"])==1
    assert inputs[1]["actual_history"]==record["actual_history"]  # context, not a source
    assert all("Earlier I claimed" not in s["text"] for s in inputs[1]["sources"])
    assert all("GOLD_MUST_NOT_APPEAR" not in json.dumps(p) and "HIDDEN_ARM" not in json.dumps(p) for p in inputs)
    claim=result["claims"][0]
    binding=claim["answer_binding"]
    assert record["answer"][binding["start"]:binding["end"]]==binding["text"]
    assert claim["visible_check"]["bound_evidence"][0]["source_spans"]==[{"start":0,"end":25}]
    assert claim["cited_check"]["verdict"]=="entailed"


@pytest.mark.asyncio
async def test_missing_citation_does_not_change_visible_fact_verdict():
    record=audit("The cache limit is 8 MiB.")
    result,requests=await run(record,extraction(record["answer"]),checked())
    claim=result["claims"][0]
    assert len(requests)==2 and claim["citation_status"]=="missing" and claim["cited_check"] is None
    assert claim["visible_check"]["verdict"]=="entailed"


@pytest.mark.asyncio
async def test_wrong_citation_cannot_use_an_uncited_visible_source():
    record=audit("The cache limit is 8 MiB [E_02].")
    result,requests=await run(record,extraction(record["answer"]),checked(),checked(
        "insufficient",quote=None))
    claim=result["claims"][0]
    assert claim["visible_check"]["verdict"]=="entailed"
    assert claim["cited_check"]["verdict"]=="insufficient"
    subset=json.loads(requests[2]["messages"][1]["content"])["sources"]
    assert [s["reference_id"] for s in subset]==["E_02"]
    assert "8 MiB" not in subset[0]["text"]


@pytest.mark.asyncio
async def test_same_source_set_reuses_check_and_keeps_joint_evidence():
    record=audit("The cache limit is 8 MiB [E_01].",sources=["The cache limit is 8 MiB."])
    result,requests=await run(record,extraction(record["answer"]),checked())
    assert len(requests)==2 and result["claims"][0]["cited_check"]["reused_visible_check"] is True


@pytest.mark.asyncio
async def test_exact_quote_never_implies_entailment():
    record=audit("The cache limit is 99 MiB.")
    result,_=await run(record,extraction(record["answer"]),checked("contradicted",relation="opposes"))
    assert result["claims"][0]["visible_check"]["verdict"]=="contradicted"
    assert result["summary"]["visible_model_verdict_counts"]["entailed"]==0


@pytest.mark.asyncio
@pytest.mark.parametrize("check",[
    checked(quote="Invented source quote"),
    checked(reference_id="E_ff"),
    checked(quote=None),
    checked(verdict="maybe"),
])
async def test_invalid_source_bindings_or_unknown_verdict_remain_failed(check):
    record=audit("The cache limit is 8 MiB.")
    result,_=await run(record,extraction(record["answer"]),check)
    response=result["claims"][0]["visible_check"]
    assert response["status"]=="failed" and "parsed" in response and response["raw_text"]
    assert result["status"]=="partial"


@pytest.mark.asyncio
async def test_checker_cannot_clean_up_extraction_and_call_it_verified():
    record=audit("The total including attachments is limited to 8 MiB.")
    result,_=await run(record,extraction(record["answer"],statement="The metadata is limited to 8 MiB."),
                       checked(faithful=False))
    assert result["claims"][0]["visible_check"]["status"]=="unresolved"
    assert result["summary"]["visible_assessed_claims"]==0


@pytest.mark.asyncio
@pytest.mark.parametrize("updates",[
    {"answer_quote":"This quote was invented."},
    {"unit_id":"u99"},
    {"citation_ids":["E_ff"]},
])
async def test_invalid_extracted_span_does_not_call_checker(updates):
    record=audit()
    result,requests=await run(record,extraction(record["answer"],**updates))
    assert len(requests)==1 and result["claims"][0]["status"]=="failed"
    assert result["coverage"]["unaccounted_units"]==["u1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["object", "relation", "value_raw", "unit_raw", "comparator", "scope",
                                  "conditions", "negation", "modality", "citation_ids"])
async def test_required_preservation_slots_cannot_be_silently_omitted(field):
    record=audit()
    out=extraction(record["answer"])
    del out["claims"][0][field]
    result,requests=await run(record,out)
    assert len(requests)==1 and result["claims"][0]["status"]=="failed"
    assert field in result["claims"][0]["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("updates", [{"conditions":"when enabled"},{"negation":1},
                                     {"object":[]},{"modality":"invented"}])
async def test_preservation_slot_structure_is_validated_once_before_check(updates):
    record=audit()
    result,requests=await run(record,extraction(record["answer"],**updates))
    assert len(requests)==1 and result["claims"][0]["status"]=="failed"


@pytest.mark.asyncio
async def test_overlapping_or_repeated_answer_span_is_not_assigned_arbitrarily():
    record=audit("ababa")
    result,requests=await run(record,extraction(record["answer"],answer_quote="aba"))
    assert len(requests)==1 and result["claims"][0]["status"]=="failed"
    assert "exactly one" in result["claims"][0]["error"]


@pytest.mark.asyncio
async def test_cited_checker_cannot_quote_a_source_only_in_visible_check():
    record=audit("The cache limit is 8 MiB [E_02].")
    result,_=await run(record,extraction(record["answer"]),checked(),checked())
    entry=result["claims"][0]
    assert entry["visible_check"]["status"]=="assessed"
    assert entry["cited_check"]["status"]=="failed"
    assert result["summary"]["visible_assessed_claims"]==1


@pytest.mark.asyncio
async def test_multiple_sources_are_bound_for_one_jointly_supported_claim():
    record=audit("The retention is 4 days [E_01, E_02].",
                 ["Retention is 96 hours.", "For this service, 24 hours equals one day."])
    check=checked(quote="Retention is 96 hours.")
    check["evidence"].append({"reference_id":"E_02", "quote":record["audit_evidence"][1]["text"], "relation":"supports"})
    result,requests=await run(record,extraction(record["answer"]),check)
    assert len(requests)==2 and len(result["claims"][0]["visible_check"]["bound_evidence"])==2


@pytest.mark.asyncio
async def test_duplicate_claim_does_not_inflate_counts_or_calls():
    record=audit("The cache limit is 8 MiB.")
    out=extraction(record["answer"])
    out["claims"].append(out["claims"][0] | {"claim_id":"c2"})
    result,requests=await run(record,out,checked())
    assert len(requests)==2 and result["summary"]["unique_bound_claims"]==1
    assert result["claims"][1]["status"]=="duplicate"


@pytest.mark.asyncio
@pytest.mark.parametrize("difference", [{"negation":True}, {"conditions":["Only when enabled"]}])
async def test_different_semantic_slots_are_not_hidden_as_duplicates(difference):
    record=audit("The cache limit is 8 MiB.")
    out=extraction(record["answer"])
    out["claims"].append(out["claims"][0] | {"claim_id":"c2"} | difference)
    result,requests=await run(record,out,checked(),checked(claim_id="c2",faithful=False))
    assert len(requests)==3 and result["summary"]["unique_bound_claims"]==2
    assert result["claims"][1]["status"]=="unresolved"
    assert result["claims"][1]["extracted"] | difference == result["claims"][1]["extracted"]
    assert result["status"]=="partial"


@pytest.mark.asyncio
async def test_code_and_uncovered_units_never_become_perfect_factuality():
    record=audit('```json\n{"enabled": true,}\n```')
    uncovered=[{"unit_id":f"u{i}","reason":"code"} for i in (1,2,3)]
    result,requests=await run(record,{"claims":[],"uncovered_units":uncovered})
    assert len(requests)==1 and result["status"]=="unassessable"
    assert result["coverage"]["structural_review_units"]==["u1","u2","u3"]
    assert result["summary"]["support_fraction"] is None


@pytest.mark.asyncio
async def test_standalone_citation_group_keeps_binding_unresolved():
    record=audit("The cache limit is 8 MiB.\n[E_01]")
    out=extraction("The cache limit is 8 MiB.")
    out["uncovered_units"]=[{"unit_id":"u2","reason":"nonverifiable"}]
    result,requests=await run(record,out,checked())
    assert len(requests)==2 and result["status"]=="partial"
    assert result["claims"][0]["visible_check"]["verdict"]=="entailed"
    assert result["claims"][0]["citation_status"]=="not_bound_to_unit"
    assert result["coverage"]["unbound_citation_units"]==["u2"]


@pytest.mark.asyncio
async def test_abstention_needs_explicit_source_answerability_judgment():
    record=audit("The visible sources do not provide a limit.")
    result,_=await run(record,extraction(record["answer"],kind="abstention"),
                       checked("not_verifiable",quote=None))
    assert result["claims"][0]["visible_check"]["status"]=="unresolved"
    result,_=await run(record,extraction(record["answer"],kind="abstention"),
                       checked("not_verifiable",quote=None,source_answers_current_question=True))
    assert result["claims"][0]["visible_check"]["parsed"]["source_answers_current_question"] is True
    assert result["summary"]["quality_score"] is None


@pytest.mark.asyncio
async def test_extraction_truncation_preserves_raw_response_and_no_checker_calls():
    record=audit()
    response=httpx.Response(200,json={"choices":[{"finish_reason":"length", "message":{
        "content":json.dumps(extraction(record["answer"]))}}],"usage":{"completion_tokens":1536}})
    result,requests=await run(record,response)
    assert len(requests)==1 and result["status"]=="failed"
    assert result["extraction"]["finish_reason"]=="length" and result["extraction"]["raw_text"]


@pytest.mark.asyncio
async def test_budget_preflight_and_exclusive_file_output(tmp_path):
    record=audit()
    audit_path=tmp_path/"audit.jsonl"
    audit_path.write_text(json.dumps(record)+"\n")
    target=tmp_path/"ledger.jsonl"
    def unexpected(_):
        pytest.fail("over-budget inputs must not be sent")
    cfg=config(context_window=2048,max_tokens=128)
    await evaluate_file(audit_path,cfg,target,transport=httpx.MockTransport(unexpected),token_counter=lambda _:2000)
    row=json.loads(target.read_text())
    assert row["status"]=="failed" and row["extraction"]["input_tokens"]==2000
    assert row["input_hash"]==token_hash([3,4])
    with pytest.raises(FileExistsError):
        await evaluate_file(audit_path,cfg,target,transport=httpx.MockTransport(unexpected),token_counter=lambda _:2000)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", [[], {}])
async def test_malformed_coverage_unit_is_recorded_and_next_file_row_runs(tmp_path,bad_id):
    record=audit("The cache limit is 8 MiB.")
    path,target=tmp_path/"audit.jsonl",tmp_path/"ledger.jsonl"
    path.write_text(json.dumps(record)+"\n"+json.dumps(record | {"task_id":"s::2"})+"\n")
    replies=Responses({"claims":[],"uncovered_units":[{"unit_id":bad_id,"reason":"nonverifiable"}]},
                      extraction(record["answer"]),checked())
    await evaluate_file(path,config(),target,transport=httpx.MockTransport(replies))
    rows=[json.loads(line) for line in target.read_text().splitlines()]
    assert len(rows)==2 and rows[0]["status"]=="failed" and rows[1]["status"]=="complete"
    assert rows[0]["coverage"]["errors"][0]["entry"]["unit_id"]==bad_id
    assert rows[1]["task_id"]=="s::2"


def test_protocol_assets_declare_adaptation_and_no_five_dimension_gate():
    root=Path(__file__).resolve().parents[1]
    p=json.loads((root/"bench/assets/atomic_judge/protocol.json").read_text())
    assert p["version"]=="contextkv-atomic-evidence-v1"
    assert p["method_provenance"]["upstream_prompt_strings_copied"] is False
    assert "not a full reproduction" in p["method_provenance"]["adaptation"]
