"""Offline Granite Guardian 4.1 native binary risk diagnostic.

Uses complete visible documents and native token IDs, not the five-dimension
judge schema. Official prompt wording is Apache-2.0; see the Granite notice.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Mapping

import httpx

from bench.judge import evaluation_binding, judge_payload
from bench.native_baseline import NativeProtocolError, NativeSGLangClient
from bench.official import load_jsonl


MODEL_ID = "ibm-granite/granite-guardian-4.1-8b"
MODEL_REVISION = "ab01ccca5dcfb80246369a086a4a87a29198f5af"
SOURCE_REVISION = "0a4ec708fa916f3e9d7fc7e5b75dff836752f448"
GROUNDEDNESS_CRITERION = (
    "A text is considered ungrounded or unfaithful if it includes information "
    "lacking support from, or directly contradicting, the provided document(s). "
    "This risk arises when the text fabricates details, misinterprets the content, "
    "or makes unsupported extrapolations beyond what is explicitly stated in the document(s)."
)
JUDGE_THINK = (
    "<think>As a judge agent, carefully analyze whether the provided text meets the "
    "given judging criteria. Use all available information, including conversations, "
    "documents, and tools. Reason through the evidence step by step before providing "
    "your score. Wrap your reasoning in <think></think> tags, then provide your "
    "score in <score></score> tags."
)
JUDGE_NOTHINK = (
    "<no-think>As a judge agent, assess whether the provided text meets the given "
    "judging criteria using all available information, including conversations, "
    "documents, and tools. Provide your score immediately without explanation. "
    "Output empty <think>\n</think> tags followed by your score in <score></score> tags."
)
SCORING_SCHEMA = "If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'."


@dataclass(frozen=True)
class GraniteJudgeConfig:
    base_url: str
    context_window: int
    model: str = MODEL_ID
    revision: str = MODEL_REVISION
    max_new_tokens: int = 64
    think: bool = False
    timeout_seconds: float = 120

    def __post_init__(self):
        if not all((self.base_url, self.model, self.revision)):
            raise ValueError("judge endpoint, model and revision must be explicit")
        if min(self.context_window, self.max_new_tokens, self.timeout_seconds) <= 0:
            raise ValueError("judge context, output budget and timeout must be positive")


def guardian_block(*, think=False):
    # The official repository helper includes <guardian>; the tokenizer does not
    # insert it. Keep the final user block separate from the answer being judged.
    return (f"<guardian>{JUDGE_THINK if think else JUDGE_NOTHINK}\n\n"
            f"### Criteria: {GROUNDEDNESS_CRITERION}\n\n### Scoring Schema: {SCORING_SCHEMA}")


def parse_score(text):
    match = re.fullmatch(r"\s*(?:<think>(.*?)</think>\s*)?<score>\s*(yes|no)\s*</score>\s*",
                         text, flags=re.DOTALL | re.IGNORECASE)
    if match is None:
        raise ValueError("expected one native <score>yes/no</score>, optionally preceded by <think>")
    value = match[2].lower()
    return {"native_score": value, "risk_detected": value == "yes",
            "reasoning": match[1].strip() if match[1] is not None else None}


def native_input(record: Mapping, *, think=False):
    payload = judge_payload(record)
    by_key = {json.dumps(block["key"], sort_keys=True): block for block in payload["audit_evidence"]}
    documents, seen = [], set()
    for reference in payload["visible_evidence_manifest"]:
        key = json.dumps(reference["key"], sort_keys=True)
        identity = (key, reference["reference_id"])
        if identity in seen:
            continue
        if key not in by_key:
            raise ValueError("visible evidence is missing its complete original body")
        documents.append({"doc_id": reference["reference_id"], "text": by_key[key]["text"]})
        seen.add(identity)
    messages = []
    for message in payload["actual_history"]:
        if message["speaker"] not in {"user", "assistant"}:
            raise ValueError("actual QA history must retain user/assistant roles")
        messages.append({"role": message["speaker"], "content": message["text"]})
    messages.extend([{"role": "user", "content": payload["query"]},
                     {"role": "assistant", "content": payload["answer"]},
                     {"role": "user", "content": guardian_block(think=think)}])
    return messages, documents


class GraniteJudge:
    """Sequential offline adapter; HTTP and token-response contracts are shared."""
    def __init__(self, config, tokenizer, client=None, *, model_context_limit=131072,
                 tokenizer_path=None):
        if config.context_window > model_context_limit:
            raise ValueError("deployed context_window exceeds the recorded model context limit")
        self.config, self.tokenizer = config, tokenizer
        self.client = client or NativeSGLangClient(config.base_url, timeout_seconds=config.timeout_seconds,
                                                  audit_prompt_ids=True, stream=False)
        self.metadata = {**asdict(config), "model_context_limit": model_context_limit,
            "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
            "revision_provenance": "operator_supplied_local_snapshot",
            "protocol": "granite_guardian_4_1_ungroundedness_native_v1",
            "source_revision": SOURCE_REVISION, "criterion": GROUNDEDNESS_CRITERION,
            "guardian_block": guardian_block(think=config.think),
            "transport": "native /generate input_ids; no session_id; no second chat template",
            "temperature": 0, "risk_polarity": "yes=ungroundedness_detected; no=not_detected",
            "evidence_policy": "all visible original bodies via documents; prior assistant turns remain conversation only",
            "quality_gate_eligible": False, "validation_status": "development_diagnostic"}

    async def evaluate(self, record):
        result = {"task_id": record["task_id"], "conversation_id": record["conversation_id"],
            "answer": record["answer"], **evaluation_binding(record),
            "source": "offline_granite_guardian", "generation_measurement_kind": record.get("measurement_kind"),
            "judge": self.metadata, "status": "failed", "native_score": None, "risk_detected": None,
            "reference_targets_supplied": False, "quality_noninferiority_established": False}
        before = len(self.client.observations)
        try:
            messages, documents = native_input(record, think=self.config.think)
            ids = list(self.tokenizer.apply_chat_template(messages, documents=documents,
                tokenize=True, add_generation_prompt=True))
            result.update(messages=messages, documents=documents, judge_input_ids=ids, judge_input_tokens=len(ids))
            if len(ids)+self.config.max_new_tokens > self.config.context_window:
                result["failure_kind"] = "input_overflow"
                raise ValueError("complete judge input plus output budget exceeds deployed context_window; nothing truncated")
            generated = await self.client.generate(ids, max_new_tokens=self.config.max_new_tokens)
            result.update(raw_text=generated.text, output_ids=list(generated.output_ids),
                finish_reason=generated.finish_reason, truncated=generated.truncated,
                meta_info=generated.meta_info, raw_response=generated.raw_response,
                http_elapsed_ms=generated.http_elapsed_ms)
            if generated.truncated:
                result["failure_kind"] = "truncated_output"
                raise ValueError("native judge output reached its token budget")
            result["failure_kind"] = "invalid_native_score"
            result.update(parse_score(generated.text))
            result.update(status="scored", failure_kind=None)
        except (httpx.HTTPError, NativeProtocolError, ValueError, KeyError, TypeError) as exc:
            result.setdefault("failure_kind", "transport_or_input_protocol")
            result["error"] = f"{type(exc).__name__}: {exc}"
        observations = self.client.observations[before:]
        if observations:
            # Native client retains error/non-JSON responses too. Keep the wire
            # artifact even if no valid NativeGeneration could be constructed.
            result["native_observation"] = observations[-1]
            result.setdefault("raw_response", observations[-1].get("response"))
            wire = observations[-1].get("response")
            if isinstance(wire, dict):
                result.setdefault("raw_text", wire.get("text"))
                result.setdefault("output_ids", wire.get("output_ids"))
                meta = wire.get("meta_info") or {}
                result.setdefault("meta_info", meta)
                reason = meta.get("finish_reason") if isinstance(meta, dict) else None
                result.setdefault("finish_reason", reason.get("type") if isinstance(reason, dict) else reason)
        return result

    async def aclose(self):
        await self.client.aclose()


async def evaluate_file(audit_path, config, output_path, *, tokenizer_path):
    from transformers import AutoConfig, AutoTokenizer
    records = load_jsonl(audit_path)
    if not records:
        raise ValueError("judge audit must not be empty")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = AutoConfig.from_pretrained(tokenizer_path, local_files_only=True)
    judge = GraniteJudge(config, tokenizer, model_context_limit=model.max_position_embeddings,
                         tokenizer_path=tokenizer_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        info = await judge.client.server_info()
        actual_window = info.get("server_args", info).get("context_length")
        if actual_window is not None and actual_window != config.context_window:
            raise ValueError("configured judge context_window differs from native server declaration")
        with target.open("x") as stream:
            for record in records:
                result = await judge.evaluate(record)
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False)+"\n")
                stream.flush()
        summary = {"rows": len(results), "scored": sum(r["status"] == "scored" for r in results),
            "failed_task_ids": [r["task_id"] for r in results if r["status"] != "scored"],
            "server_info": info, "protocol": judge.metadata, "quality_accepted": False,
            "interpretation": "binary ungroundedness risk diagnostic; no five-dimensional overall score"}
        with target.with_suffix(".summary.json").open("x") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
        return summary
    finally:
        await judge.aclose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    args = parser.parse_args(argv)
    config = GraniteJudgeConfig(**json.loads(Path(args.config).read_text()))
    summary = asyncio.run(evaluate_file(args.audit, config, args.output, tokenizer_path=args.tokenizer_path))
    print(json.dumps({k: v for k, v in summary.items() if k not in {"server_info", "protocol"}}, indent=2))
    if summary["failed_task_ids"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
