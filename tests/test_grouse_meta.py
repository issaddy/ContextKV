"""GroUSE native JSON/conditional contracts, with fake calls only; no model scores."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import httpx
import pytest

from bench import grouse_meta as gm


def sample(**conditions):
    return {"input":"USER_QUESTION", "actual_output":"CANDIDATE_ANSWER", "expected_output":"REFERENCE_ANSWER",
            "references":["SOURCE_FULL_BODY {not_a_template}","SECOND_SOURCE"],
            "metadata":{"goal":"DO_NOT_SEND_GOAL_928","test_type":"DO_NOT_SEND_TYPE_346"},
            "conditions":{metric+"_condition":conditions.get(metric,"==None") for metric in gm.DIRECT}}


def pair(metric,value,first=1):
    def answer(score):
        result={metric:score,metric+"_justification":"because"}
        if metric=="answer_relevancy":
            result["answer_affirms_no_document_answers"]=score is None
        return result
    return json.dumps({"answer_1":answer(first),"answer_2":answer(value)})


class Calls:
    def __init__(self, responses):
        self.responses=iter(responses)
        self.messages=[]
    async def __call__(self,messages):
        self.messages.append(messages)
        value=next(self.responses)
        if isinstance(value,Exception):
            raise value
        return value


@pytest.mark.parametrize("values,metrics,acceptance", [
    ((5,4,1),("answer_relevancy","completeness","faithfulness"),(None,None)),
    ((None,None,None),("answer_relevancy","completeness","usefulness"),(1,1)),
    ((None,3,0,0),("answer_relevancy","completeness","usefulness","faithfulness"),(0,None)),
])
def test_official_three_or_four_call_flow_and_null_derivation(values,metrics,acceptance):
    call=Calls([pair(metric,value) for metric,value in zip(metrics,values)])
    evaluation=asyncio.run(gm.GroUSEEvaluator(call).evaluate_sample(sample()))
    assert [c["metric"] for c in evaluation["calls"]]==list(metrics)
    assert tuple(evaluation["scores"][m]["value"] for m in gm.METRICS[-2:])==acceptance
    if "faithfulness" not in metrics:
        assert evaluation["scores"]["faithfulness"]==gm.score(None,skipped=True)
    if "usefulness" not in metrics:
        assert evaluation["scores"]["usefulness"]==gm.score(None,skipped=True)
    prompts="\n".join(m[0]["content"] for m in call.messages)
    assert "DO_NOT_SEND_GOAL_928" not in prompts and "DO_NOT_SEND_TYPE_346" not in prompts
    assert "SOURCE_FULL_BODY {not_a_template}" in prompts and "SECOND_SOURCE" in prompts
    assert "Answer 1: REFERENCE_ANSWER" in prompts and "Answer 2: CANDIDATE_ANSWER" in prompts


def test_parser_reads_second_answer_and_matches_fenced_json_contract():
    result=gm.parse_pair("comment\n```json\n"+pair("answer_relevancy",2,first=5)+"\n```","answer_relevancy")
    assert result["value"]==2
    omitted=json.loads(pair("completeness",None))
    del omitted["answer_2"]["completeness"]
    assert gm.parse_pair(json.dumps(omitted),"completeness")["value"] is None


@pytest.mark.parametrize("value,expected",[(True,1),(False,0),(None,None)])
def test_official_faithfulness_boolean_or_null_is_normalized_like_upstream_dto(value,expected):
    assert gm.parse_pair(pair("faithfulness",value),"faithfulness")["value"]==expected


@pytest.mark.parametrize("metric,value",[("answer_relevancy",0),("completeness",6),("faithfulness",2),
                                         ("usefulness",True),("usefulness","1")])
def test_values_outside_template_protocol_fail_instead_of_earning_meta_credit(metric,value):
    with pytest.raises(ValueError,match="must be null"):
        gm.parse_pair(pair(metric,value),metric)


@pytest.mark.parametrize("value,expected,rule",[("null",None,"string_null"),("+1",1,"integer_string"),
    ("01",1,"integer_string"),(1.0,1,"integer_float")])
def test_optional_normalization_converts_only_score_fields_and_records_exact_change(value,expected,rule):
    raw=pair("completeness",value)
    trace=[]
    with pytest.raises(ValueError):
        gm.parse_pair(raw,"completeness")  # default strict_v1 preserves its original acceptance
    result=gm.parse_pair(raw,"completeness",parser_mode="normalized_v1",normalization_trace=trace)
    assert result["value"]==expected and result["justification"]=="because"
    assert trace==[{"answer":"answer_2","metric":"completeness","rule":rule,
                   "original":value,"normalized":expected}]


@pytest.mark.parametrize("metric,value",[("completeness","NULL"),("completeness"," null "),
    ("completeness","1.0"),("completeness","1e0"),("completeness"," 1 "),
    ("completeness","１"),("completeness",1.5),("completeness",float("inf")),
    ("completeness",float("nan")),("completeness","6"),("completeness",True),
    ("usefulness",True),("usefulness","2"),("faithfulness","true")])
def test_optional_normalization_retains_ranges_boolean_rules_and_unambiguous_grammar(metric,value):
    with pytest.raises(ValueError,match="must be null"):
        gm.parse_pair(pair(metric,value),metric,parser_mode="normalized_v1")


def test_optional_normalization_does_not_coerce_justifications_or_other_flags():
    value=json.loads(pair("answer_relevancy","1"))
    value["answer_2"]["answer_affirms_no_document_answers"]="false"
    with pytest.raises(ValueError,match="boolean"):
        gm.parse_pair(json.dumps(value),"answer_relevancy",parser_mode="normalized_v1")
    value=json.loads(pair("completeness","1"))
    value["answer_2"]["completeness_justification"]=1
    with pytest.raises(ValueError,match="justification string"):
        gm.parse_pair(json.dumps(value),"completeness",parser_mode="normalized_v1")


def test_normalized_mode_records_changes_and_preserves_raw_when_pair_still_fails():
    raw=pair("answer_relevancy",6,first="null")
    call=Calls([raw,pair("completeness",None)])
    result=asyncio.run(gm.GroUSEEvaluator(call,parser_mode="normalized_v1").evaluate_sample(sample()))
    record=result["calls"][0]
    assert record["raw_response"]==raw and record["status"]=="failed"
    assert record["normalization_trace"]==[{"answer":"answer_1","metric":"answer_relevancy",
        "rule":"string_null","original":"null","normalized":None}]
    assert result["scores"]["usefulness"]["status"]=="failed" and len(call.messages)==2


def test_normalized_config_runs_all_new_required_calls_and_freeze_binds_mode(tmp_path,monkeypatch):
    monkeypatch.setattr(gm,"load_split",lambda split,**kwargs:([sample()],{"split":split,"rows":1}))
    config=gm.GrouseConfig("model","revision","http://test/v1",parser_mode="normalized_v1")
    raw=[pair("answer_relevancy","null",first="null"),pair("completeness","null"),
         pair("usefulness","1"),pair("faithfulness",False)]
    call=Calls(raw)
    run=tmp_path/"train-normalized"
    report=asyncio.run(gm.evaluate_run(config,"train",run,call=call))
    assert report["calls"]==4 and report["failed_calls"]==0
    result=json.loads((run/"results.jsonl").read_text())
    assert [c["raw_response"] for c in result["evaluation"]["calls"]]==raw
    assert result["evaluation"]["calls"][-1]["normalization_trace"]==[
        {"answer":"answer_2","metric":"faithfulness","rule":"faithfulness_boolean",
         "original":False,"normalized":0}]
    frozen=tmp_path/"frozen-normalized.json"
    gm.freeze(run,frozen)
    assert gm.resolve_evaluator("test",frozen_path=frozen)==config
    identity=json.loads(frozen.read_text())["evaluator"]
    assert identity["config"]["parser_mode"]=="normalized_v1"
    assert identity["parser"]=="bounded-normalized-json-scores-v1"
    value=json.loads(frozen.read_text())
    value["evaluator"]["config"]["parser_mode"]="strict_v1"
    frozen.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="differs"):
        gm.resolve_evaluator("test",frozen_path=frozen)


def test_unknown_parser_mode_rejected_at_configuration_boundary():
    with pytest.raises(ValueError,match="parser_mode"):
        gm.GrouseConfig("model","revision","http://test/v1",parser_mode="guess")


def test_failed_relevancy_keeps_completeness_call_and_propagates_failed_not_null():
    call=Calls(["invalid JSON",pair("completeness",5)])
    row=sample(completeness="==5")
    evaluation=asyncio.run(gm.GroUSEEvaluator(call).evaluate_sample(row))
    assert len(call.messages)==2
    assert evaluation["scores"]["completeness"]["value"]==5
    assert all(evaluation["scores"][m]["status"]=="failed" for m in gm.METRICS if m!="completeness")
    record={"evaluation":evaluation,"condition_results":gm.evaluate_conditions(evaluation,row["conditions"])}
    report=gm.summarize([record])
    assert report["samples"]==1 and report["total"]==pytest.approx(1/6)
    assert report["metric_success"]["usefulness"]==0  # must not pass expected null
    assert report["failed_calls"]==1 and report["calls"]==2
    assert report["usage_totals"]["total_tokens"] is None


def test_failed_completeness_does_not_skip_independent_faithfulness():
    call=Calls([pair("answer_relevancy",5),httpx.ReadTimeout("offline fake"),pair("faithfulness",1)])
    evaluation=asyncio.run(gm.GroUSEEvaluator(call).evaluate_sample(sample()))
    assert len(call.messages)==3 and evaluation["scores"]["faithfulness"]["value"]==1
    assert evaluation["scores"]["positive_acceptance"]["status"]=="failed"


def test_usefulness_failure_prevents_faithfulness_but_does_not_invalidate_acceptance():
    call=Calls([pair("answer_relevancy",None),pair("completeness",None),"bad"])
    evaluation=asyncio.run(gm.GroUSEEvaluator(call).evaluate_sample(sample()))
    assert len(call.messages)==3 and evaluation["scores"]["faithfulness"]["status"]=="failed"
    assert evaluation["scores"]["positive_acceptance"]["value"]==1


@pytest.mark.parametrize("r,c,expected",[(None,None,(1,1)),(None,3,(0,None)),(3,None,(None,0)),(3,3,(None,None))])
def test_all_nullable_acceptance_states(r,c,expected):
    result=gm.derive_acceptance(gm.score(r),gm.score(c))
    assert tuple(s["value"] for s in result)==expected


@pytest.mark.parametrize("value,condition,passed",[(None,"==None",True),(1,"==None",False),(None,">=3",False),
    (3,">=3",True),(3,">3",False),(2,"<3",True),(2,"<=2",True),(1,"==1",True)])
def test_offline_comparison_operators(value,condition,passed):
    assert gm.compare(value,condition) is passed


@pytest.mark.parametrize("condition",["__import__('os')","!=1",">=None","==NaN"])
def test_unknown_condition_is_rejected_before_use(condition):
    with pytest.raises(ValueError,match="unsupported GroUSE condition"):
        gm.parse_condition(condition)


@pytest.mark.parametrize("finish_reason", ["length",None,"content_filter"])
def test_http_transport_retains_usage_and_incomplete_valid_json_is_failed(finish_reason):
    async def scenario():
        sent=[]
        def handler(request):
            body=json.loads(request.content)
            sent.append(body)
            metric="answer_relevancy" if len(sent)==1 else "completeness"
            return httpx.Response(200,json={"model":"served-model","id":"fake",
                "usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17},
                "choices":[{"finish_reason":finish_reason if len(sent)==1 else "stop",
                            "message":{"content":pair(metric,5)}}]})
        config=gm.GrouseConfig("model","fixed-revision","http://test/v1")
        call=gm.HTTPCall(config,transport=httpx.MockTransport(handler))
        try:
            result=await gm.GroUSEEvaluator(call).evaluate_sample(sample())
            assert result["scores"]["answer_relevancy"]["status"]=="failed"
            assert "incomplete judge output" in result["scores"]["answer_relevancy"]["error"]
            assert result["calls"][0]["usage"]["total_tokens"]==17
            assert sent[0]["model"]=="model" and sent[0]["temperature"]==.01
            assert sent[0]["max_tokens"]==2048 and sent[0]["stream"] is False
            assert "response_format" not in sent[0]
        finally:
            await call.aclose()
    asyncio.run(scenario())


def test_fixed_single_split_loader_verifies_local_file_without_loading_other_split(monkeypatch,tmp_path):
    raw=(json.dumps(sample())+"\n").encode()
    manifest=deepcopy(gm.source_manifest())
    manifest["dataset"]["files"]["train.jsonl"]["oid"]=hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
    monkeypatch.setattr(gm,"source_manifest",lambda:manifest)
    urls=[]
    def fetch(url):
        urls.append(url)
        assert url.endswith("/train.jsonl")
        return raw
    rows,binding=gm.load_split("train",fetch=fetch)
    assert len(rows)==1 and len(urls)==1
    assert "/caaaae0dacf5126279ae59609d26926e124b8032/" in urls[0]
    assert binding["split"]=="train"
    p=tmp_path/"train.jsonl"
    p.write_bytes(raw)
    assert gm.load_split("train",dataset_file=p)[0]==rows
    p.write_bytes(raw+b"\n")
    with pytest.raises(ValueError,match="pinned split"):
        gm.load_split("train",dataset_file=p)


def test_train_freeze_test_order_and_failed_rows_are_preserved(tmp_path,monkeypatch):
    row=sample(answer_relevancy="==5",completeness="==5",faithfulness="==1")
    loaded=[]
    def load(split,**kwargs):
        loaded.append(split)
        return [row,row],{"split":split,"revision":"fixed-test-fixture","rows":2}
    monkeypatch.setattr(gm,"load_split",load)
    config=gm.GrouseConfig("model","revision","http://test/v1",provider="test")
    with pytest.raises(ValueError,match="test requires"):
        asyncio.run(gm.evaluate_run(config,"test",tmp_path/"forbidden",call=Calls([])))
    assert loaded==[] and not (tmp_path/"forbidden").exists()
    call=Calls([pair("answer_relevancy",5),pair("completeness",5),pair("faithfulness",1),"bad",pair("completeness",5)])
    output=tmp_path/"train"
    report=asyncio.run(gm.evaluate_run(config,"train",output,call=call))
    assert loaded==["train"] and report["samples"]==2
    rows=[json.loads(line) for line in (output/"results.jsonl").read_text().splitlines()]
    assert len(rows)==2 and rows[1]["evaluation"]["scores"]["answer_relevancy"]["status"]=="failed"
    assert json.loads((output/"manifest.json").read_text())["completed_samples"]==2
    frozen=tmp_path/"frozen.json"
    gm.freeze(output,frozen)
    assert gm.resolve_evaluator("test",frozen_path=frozen)==config
    untouched=Calls([])
    with pytest.raises(FileExistsError):
        asyncio.run(gm.evaluate_run(config,"train",output,call=untouched))
    assert untouched.messages==[]
    value=json.loads(frozen.read_text())
    value["evaluator"]["parser"]="different-parser"
    frozen.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="differs"):
        gm.resolve_evaluator("test",frozen_path=frozen)
    assert loaded==["train"]  # frozen checks never read test data


def test_changed_bundled_template_and_incomplete_train_cannot_freeze(tmp_path,monkeypatch):
    target=tmp_path/"assets"
    shutil.copytree(gm.ASSETS,target)
    monkeypatch.setattr(gm,"ASSETS",target)
    p=target/"faithfulness.txt.jinja"
    p.write_text(p.read_text()+"changed")
    with pytest.raises(ValueError,match="pinned source"):
        gm.templates()
    run=tmp_path/"run"
    run.mkdir()
    (run/"manifest.json").write_text(json.dumps({"status":"incomplete","dataset":{"split":"train"}}))
    with pytest.raises(ValueError,match="completed train"):
        gm.freeze(run,tmp_path/"frozen.json")


def test_no_evaluation_dependency_is_installed_implicitly(monkeypatch):
    import builtins
    original=builtins.__import__
    def importing(name,*args,**kwargs):
        if name=="jinja2":
            raise ImportError("optional dependency absent")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",importing)
    with pytest.raises(RuntimeError,match="evaluation"):
        gm.templates()


def test_source_license_and_prompts_are_bundled_with_fixed_origin():
    manifest=gm.source_manifest()
    assert manifest["commit"]=="e4c57ba9eb14548c7642daca3ed4ff24d760c7fa"
    assert "Copyright (c) 2024 Sacha Muller, Antonio Loison, Bilel Omrani" in (gm.ASSETS/"LICENSE.txt").read_text()
    assert len(gm.templates())==4
    assert Path(gm.__file__).read_text().find("LiteLLM")<0
