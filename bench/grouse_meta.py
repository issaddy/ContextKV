"""GroUSE four-template JSON meta-evaluation over a plain HTTP judge endpoint.

Conditional flow and acceptance/rejection rules adapted from MIT-licensed GroUSE:
Copyright (c) 2024 Sacha Muller, Antonio Loison, Bilel Omrani.
See assets/grouse/LICENSE.txt and source_manifest.json for the pinned sources.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import re
import time
from typing import Awaitable, Callable, Mapping

import httpx

ASSETS = Path(__file__).parent / "assets/grouse"
PROTOCOL = "grouse-four-template-json-v1"
PARSER = "bounded-json-scores-v1"
PARSERS = {"strict_v1":PARSER,"normalized_v1":"bounded-normalized-json-scores-v1"}
DIRECT = ("answer_relevancy", "completeness", "faithfulness", "usefulness")
METRICS = (*DIRECT, "positive_acceptance", "negative_rejection")
SAMPLE_FIELDS = ("input", "actual_output", "expected_output", "references")
OPERATORS = {">=":operator.ge,"<=":operator.le,">":operator.gt,"<":operator.lt,"==":operator.eq}
Call = Callable[[list[dict[str, str]]], Awaitable[str | Mapping]]


@dataclass(frozen=True)
class GrouseConfig:
    model: str
    revision: str
    base_url: str
    provider: str = "local_sglang"
    temperature: float = 0.01
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    concurrency: int = 1
    api_key_env: str = "CONTEXTKV_JUDGE_API_KEY"
    parser_mode: str = "strict_v1"

    def __post_init__(self):
        if not all((self.model,self.revision,self.base_url,self.provider)):
            raise ValueError("model/revision/base_url/provider must be explicit")
        if self.max_tokens<1 or self.concurrency<1 or self.timeout_seconds<=0 or self.temperature<0:
            raise ValueError("invalid judge sampling configuration")
        if self.parser_mode not in PARSERS:
            raise ValueError(f"unknown GroUSE parser_mode: {self.parser_mode}")


class HTTPCall:
    """No response_format override: execute the frozen GroUSE JSON instructions."""
    def __init__(self, config: GrouseConfig, *, transport=None):
        self.config=config
        key=os.environ.get(config.api_key_env)
        self.client=httpx.AsyncClient(base_url=config.base_url.rstrip("/")+"/",
            headers={"Authorization":f"Bearer {key}"} if key else {},
            timeout=config.timeout_seconds,transport=transport)

    async def __call__(self, messages):
        response=await self.client.post("chat/completions",json={"model":self.config.model,
            "temperature":self.config.temperature,"max_tokens":self.config.max_tokens,
            "stream":False,"messages":messages})
        response.raise_for_status()
        return response.json()

    async def aclose(self):
        await self.client.aclose()


def source_manifest():
    return json.loads((ASSETS/"source_manifest.json").read_text())


def templates():
    try:
        from jinja2 import Environment, StrictUndefined
    except ImportError as error:
        raise RuntimeError("GroUSE templates require the evaluation extra: pip install '.[evaluation]'") from error
    manifest=source_manifest()
    environment=Environment(undefined=StrictUndefined)
    result={}
    for metric in DIRECT:
        name=f"{metric}.txt.jinja"
        raw=(ASSETS/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest["files"][name]["sha256"]:
            raise ValueError(f"bundled GroUSE template differs from pinned source: {name}")
        result[metric]=environment.from_string(raw.decode())
    return result


def evaluator_identity(config):
    manifest=source_manifest()
    return {"protocol":PROTOCOL,"parser":PARSERS[config.parser_mode],"config":asdict(config),
        "evaluator_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_commit":manifest["commit"],"dataset_revision":manifest["dataset"]["revision"],
        "template_sha256":{name:entry["sha256"] for name,entry in manifest["files"].items() if name.endswith(".jinja")},
        "reference_policy":"expected_output is answer_1; actual_output is answer_2",
        "offline_only_fields":["conditions","metadata"],"model_revision_provenance":"operator_declared"}


def failed(error):
    return {"status":"failed","value":None,"error":str(error)}


def score(value, justification="", *, skipped=False):
    return {"status":"ok","value":value,"justification":justification,"skipped":skipped}


def _normalize_score(value):
    if value == "null":
        return None,"string_null"
    if isinstance(value,str) and re.fullmatch(r"[+-]?[0-9]+",value):
        return int(value),"integer_string"
    if type(value) is float and math.isfinite(value) and value.is_integer():
        return int(value),"integer_float"
    return value,None


def parse_pair(text, metric, *, parser_mode="strict_v1", normalization_trace=None):
    if parser_mode not in PARSERS:
        raise ValueError(f"unknown GroUSE parser_mode: {parser_mode}")
    # Preserve upstream fenced-JSON handling and require both answer objects.
    matches=re.findall(r"```[a-zA-Z]*\n(.*?)```",text,re.DOTALL)
    pair=json.loads(matches[0].strip() if matches else text.strip())
    if not isinstance(pair,dict):
        raise ValueError("response must be a JSON object")
    for name in ("answer_1","answer_2"):
        item=pair[name]
        if not isinstance(item,dict) or not isinstance(item.get(metric+"_justification"),str):
            raise ValueError(f"{name} requires a {metric}_justification string")
        if metric=="answer_relevancy" and type(item.get("answer_affirms_no_document_answers")) is not bool:
            raise ValueError(f"{name} requires answer_affirms_no_document_answers boolean")
        # Upstream DTO defaults omitted score fields to None. Enforce the
        # template's stated numeric range, which its permissive DTO does not.
        value=item.get(metric)
        if parser_mode=="normalized_v1":
            original=value
            value,rule=_normalize_score(value)
            if rule is not None:
                item[metric]=value
                if normalization_trace is not None:
                    normalization_trace.append({"answer":name,"metric":metric,"rule":rule,
                                                "original":original,"normalized":value})
        if metric=="faithfulness" and type(value) is bool:
            # Its official template explicitly requests boolean/null; upstream
            # Pydantic Optional[int] normalizes true/false to one/zero.
            value=item[metric]=int(value)
            if normalization_trace is not None:
                normalization_trace.append({"answer":name,"metric":metric,"rule":"faithfulness_boolean",
                                            "original":bool(value),"normalized":value})
        allowed=range(1,6) if metric in DIRECT[:2] else range(2)
        if value is not None and (type(value) is not int or value not in allowed):
            raise ValueError(f"{metric} must be null or an integer in {list(allowed)}")
    item=pair["answer_2"]
    return score(item.get(metric),item[metric+"_justification"])


def derive_acceptance(relevancy, completeness):
    if relevancy["status"]=="failed":
        return failed("answer_relevancy failed"),failed("answer_relevancy failed")
    if completeness["status"]=="failed":
        return failed("completeness failed"),failed("completeness failed")
    r,c=relevancy["value"],completeness["value"]
    values=(1,1) if r is None and c is None else (0,None) if r is None else (None,0) if c is None else (None,None)
    return tuple(score(value) for value in values)


class GroUSEEvaluator:
    def __init__(self, call: Call, *, parser_mode="strict_v1"):
        self.call=call
        if parser_mode not in PARSERS:
            raise ValueError(f"unknown GroUSE parser_mode: {parser_mode}")
        self.parser_mode=parser_mode
        self.templates=templates()

    def messages(self, metric, sample):
        # Never pass the whole dataset row to the template environment.
        fields={name:sample[name] for name in SAMPLE_FIELDS}
        fields["contexts"]=fields.pop("references")
        return [{"role":"user","content":self.templates[metric].render(**fields)}]

    async def evaluate_sample(self, sample):
        calls=[]
        async def invoke(metric):
            messages=self.messages(metric,sample)
            started=time.perf_counter()
            record={"metric":metric,"raw_response":None,"normalization_trace":[]}
            try:
                response=await self.call(messages)
                record["raw_response"]=response
                if isinstance(response,str):
                    text=response
                else:
                    choice=response["choices"][0]
                    record.update(finish_reason=choice.get("finish_reason"),usage=response.get("usage"),
                                  served_model=response.get("model"),response_id=response.get("id"))
                    if choice.get("finish_reason")!="stop":
                        raise ValueError(f"incomplete judge output: {choice.get('finish_reason')}")
                    text=choice["message"]["content"]
                value=parse_pair(text,metric,parser_mode=self.parser_mode,
                                 normalization_trace=record["normalization_trace"])
                record["status"]="ok"
            except (httpx.HTTPError,ValueError,TypeError,KeyError,IndexError) as error:
                value=failed(f"{type(error).__name__}: {error}")
                record.update(status="failed",error=value["error"])
            record["elapsed_ms"]=(time.perf_counter()-started)*1000
            calls.append(record)
            return value
        relevancy=await invoke("answer_relevancy")
        completeness=await invoke("completeness")
        if relevancy["status"]=="failed":
            usefulness,faithfulness=failed("answer_relevancy failed"),failed("answer_relevancy failed")
        elif relevancy["value"] is None:
            usefulness=await invoke("usefulness")
            if usefulness["status"]=="failed":
                faithfulness=failed("usefulness failed")
            elif usefulness["value"] is None:
                faithfulness=score(None,skipped=True)
            else:
                faithfulness=await invoke("faithfulness")
        else:
            usefulness=score(None,skipped=True)
            faithfulness=await invoke("faithfulness")
        positive,negative=derive_acceptance(relevancy,completeness)
        return {"scores":dict(zip(METRICS,(relevancy,completeness,faithfulness,usefulness,positive,negative))),"calls":calls}


def parse_condition(condition):
    if condition=="==None":
        return None,None
    match=re.fullmatch(r"(>=|<=|==|>|<)([+-]?(?:\d+(?:\.\d*)?|\.\d+))",condition)
    if not match:
        raise ValueError(f"unsupported GroUSE condition: {condition}")
    return OPERATORS[match[1]],float(match[2])


def compare(value, condition):
    op,target=parse_condition(condition)
    if value is None:
        return op is None
    return op(value,target) if op is not None else False


def evaluate_conditions(evaluation, conditions):
    expected_r=score(None if conditions["answer_relevancy_condition"]=="==None" else 1)
    expected_c=score(None if conditions["completeness_condition"]=="==None" else 1)
    positive,negative=derive_acceptance(expected_r,expected_c)
    expected={metric:conditions[metric+"_condition"] for metric in DIRECT}
    expected.update(positive_acceptance="=="+str(positive["value"]),negative_rejection="=="+str(negative["value"]))
    return {metric:{"status":evaluation["scores"][metric]["status"],
        "passed":False if evaluation["scores"][metric]["status"]=="failed" else compare(evaluation["scores"][metric]["value"],expected[metric])}
        for metric in METRICS}


def summarize(rows):
    count=len(rows)
    success={metric:sum(r["condition_results"][metric]["passed"] for r in rows)/count if count else None for metric in METRICS}
    calls=[c for row in rows for c in row["evaluation"]["calls"]]
    return {"samples":count,"metric_success":success,"total":sum(success.values())/len(METRICS) if count else None,
        "failed_metrics":{metric:sum(r["condition_results"][metric]["status"]=="failed" for r in rows) for metric in METRICS},
        "calls":len(calls),"failed_calls":sum(c["status"]=="failed" for c in calls),
        "called_parse_success_rate":sum(c["status"]=="ok" for c in calls)/len(calls) if calls else None,
        "call_elapsed_ms_sum":sum(c["elapsed_ms"] for c in calls),
        "usage_available_calls":{key:sum((c.get("usage") or {}).get(key) is not None for c in calls)
                                  for key in ("prompt_tokens","completion_tokens","total_tokens")},
        "usage_totals":{key:sum(c["usage"][key] for c in calls if (c.get("usage") or {}).get(key) is not None)
                        if any((c.get("usage") or {}).get(key) is not None for c in calls) else None
                        for key in ("prompt_tokens","completion_tokens","total_tokens")},
        "denominator_policy":"all selected split rows; Failed contributes zero, never deleted"}


def load_split(split, *, dataset_file=None, fetch=None):
    """Read exactly one pinned file; test metadata is not test sample access."""
    source=source_manifest()["dataset"]
    filename=source["splits"][split]
    url=f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/{filename}"
    if dataset_file:
        raw=Path(dataset_file).read_bytes()
    elif fetch:
        raw=fetch(url)
    else:
        response=httpx.get(url,follow_redirects=True,timeout=60)
        response.raise_for_status()
        raw=response.content
    blob=hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
    if blob!=source["files"][filename]["oid"]:
        raise ValueError("dataset file does not match the pinned split Git blob")
    rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    for row in rows:
        for key in SAMPLE_FIELDS:
            if key not in row:
                raise ValueError(f"dataset row lacks {key}")
        for metric in DIRECT:
            parse_condition(row["conditions"][metric+"_condition"])
    return rows,{"repository":source["repository"],"revision":source["revision"],"split":split,
                 "url":url,"git_blob_sha1":blob,"rows":len(rows)}


def resolve_evaluator(split, *, config_path=None, frozen_path=None):
    if split=="test" and not frozen_path:
        raise ValueError("test requires --frozen-evaluator produced from a completed train run")
    if frozen_path:
        frozen=json.loads(Path(frozen_path).read_text())
        config=GrouseConfig(**frozen["evaluator"]["config"])
        if frozen.get("frozen_from_split")!="train" or frozen["evaluator"]!=evaluator_identity(config):
            raise ValueError("frozen evaluator differs from the current protocol/templates")
    else:
        config=GrouseConfig(**json.loads(Path(config_path).read_text()))
    # Verify the actual template bytes before any dataset/network operation.
    templates()
    return config


async def evaluate_run(config, split, output, *, dataset_file=None, frozen_path=None, call=None):
    identity=evaluator_identity(config)
    if split=="test":
        expected=resolve_evaluator(split,frozen_path=frozen_path)
        if expected!=config:
            raise ValueError("test configuration differs from frozen evaluator")
    target=Path(output)
    target.mkdir(parents=True,exist_ok=False)
    manifest={"status":"started","evaluator":identity,"mode":"train_development" if split=="train" else "frozen_protocol_test",
              "frozen_evaluator_path":str(frozen_path) if frozen_path else None}
    rows=[]
    http=None
    try:
        if call is None:
            http=HTTPCall(config)
            call=http
        evaluator=GroUSEEvaluator(call,parser_mode=config.parser_mode)
        inputs,dataset=load_split(split,dataset_file=dataset_file)
        manifest["dataset"]=dataset
        (target/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        semaphore=asyncio.Semaphore(config.concurrency)
        with (target/"results.jsonl").open("x") as stream:
            async def evaluate(index,row):
                async with semaphore:
                    result=await evaluator.evaluate_sample(row)
                record={"case_id":f"{split}:{index}","sample":{k:row[k] for k in SAMPLE_FIELDS},
                    "offline_conditions":row["conditions"],"offline_metadata":row.get("metadata"),
                    "evaluation":result,"condition_results":evaluate_conditions(result,row["conditions"])}
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
                stream.flush()
                rows.append(record)
                return record
            await asyncio.gather(*(evaluate(i,row) for i,row in enumerate(inputs)))
        report=summarize(rows)
        (target/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        manifest["status"]="complete"
        return report
    finally:
        if manifest["status"]!="complete":
            manifest["status"]="incomplete"
        manifest["completed_samples"]=len(rows)
        (target/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        if http is not None:
            await http.aclose()


def freeze(train_run, output):
    manifest=json.loads((Path(train_run)/"manifest.json").read_text())
    if manifest.get("status")!="complete" or manifest.get("dataset",{}).get("split")!="train":
        raise ValueError("freeze requires a completed train-development run")
    identity=manifest["evaluator"]
    config=GrouseConfig(**identity["config"])
    if identity!=evaluator_identity(config):
        raise ValueError("train run belongs to a different evaluator protocol")
    templates()
    target=Path(output)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x") as stream:
        json.dump({"evaluator":identity,"frozen_from_split":"train","train_run":str(train_run),
                   "train_dataset":manifest["dataset"],"frozen_protocol_not_quality_acceptance":True},stream,indent=2)
        stream.write("\n")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    run=sub.add_parser("evaluate")
    run.add_argument("--split",choices=("train","test"),default="train")
    config=run.add_mutually_exclusive_group(required=True)
    config.add_argument("--config")
    config.add_argument("--frozen-evaluator")
    run.add_argument("--dataset-file",help="optional exact pinned split file, verified before calls")
    run.add_argument("--output",required=True,help="new run directory; existing outputs are never overwritten")
    frozen=sub.add_parser("freeze")
    frozen.add_argument("--train-run",required=True)
    frozen.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    if args.command=="freeze":
        freeze(args.train_run,args.output)
    else:
        config=resolve_evaluator(args.split,config_path=args.config,frozen_path=args.frozen_evaluator)
        result=asyncio.run(evaluate_run(config,args.split,args.output,dataset_file=args.dataset_file,
                                      frozen_path=args.frozen_evaluator))
        print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
