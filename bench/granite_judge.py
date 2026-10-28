"""Offline Granite Guardian 4.1 native binary risk diagnostic.

Uses complete documents and native token IDs, not the five-dimension judge
schema. Evidence profiles distinguish legacy visibility from current authority.
Official prompt wording is Apache-2.0; see the Granite notice.
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

from bench.judge import evaluation_binding, judge_payload, rule_audit
from bench.native_baseline import NativeProtocolError, NativeSGLangClient
from bench.official import load_jsonl
from controller.schema import EvidenceBlock
from controller.serialization import decode
from controller.templates import _evidence_header


MODEL_ID = "ibm-granite/granite-guardian-4.1-8b"
MODEL_REVISION = "ab01ccca5dcfb80246369a086a4a87a29198f5af"
SOURCE_REVISION = "0a4ec708fa916f3e9d7fc7e5b75dff836752f448"
EVIDENCE_PROFILES = {
    "visible_text_v1": "all visible original bodies; legacy diagnostic",
    "current_text_v1": "complete original bodies whose IDs are in the current citation_scope",
    "current_provenance_v1": (
        "current citation_scope bodies with the generator's source/title/revision header; "
        "original body text is unchanged"
    ),
}
GROUNDEDNESS_CRITERION = (
    "A text is considered ungrounded or unfaithful if it includes information "
    "lacking support from, or directly contradicting, the provided document(s). "
    "This risk arises when the text fabricates details, misinterprets the content, "
    "or makes unsupported extrapolations beyond what is explicitly stated in the document(s)."
)
CRITERION_PROFILES = {
    "ungroundedness": GROUNDEDNESS_CRITERION,
    "relation_scope_v1": (
        "The last assistant's text has a source-grounding risk if any factual assertion lacks support "
        "from, or contradicts, the provided documents when its complete relation is considered. "
        "Preserve the object, value, unit, comparator, per-item versus aggregate scope, inclusion or "
        "exclusion, negation, and explicit conditions, exceptions, versions and roles together. "
        "A value supported for a different object or condition does not support the assertion. "
        "Separate table rows do not by themselves establish containment, addition, or a performance "
        "guarantee. Apply explicit source conditions and valid deductions from supplied relations, "
        "including unit conversions; do not require a supported deduction to be stated verbatim. "
        "Conversation history may resolve references, but prior assistant assertions are not source "
        "evidence. Assess a statement that the sources do not answer the current question against "
        "all provided documents: a justified evidence-limited abstention is not a risk, whereas "
        "denying an answer explicitly available in the sources is a risk. Judge the current answer, "
        "not the truth of every previous conversation turn."
    ),
    "citation_binding_v1": (
        "The last assistant's text has a citation-binding risk if a cited evidence identifier has "
        "no matching provided document, or if the document or documents explicitly cited for a "
        "factual assertion do not support that assertion with its object, value, unit, scope and "
        "conditions. Match bracketed evidence identifiers to document doc_id values. A comma-separated "
        "group cites its listed documents jointly; combine them only when their contents support "
        "the asserted relation. Support available only in an uncited document does not repair the "
        "citation. Prior assistant assertions are not source evidence. Evaluate the actual citations "
        "in the current answer, not citations in conversation history. Do not infer that uncited "
        "assertions are correct or incorrect, and do not treat citation presence alone as support."
    ),
}
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
    criterion_profile: str = "ungroundedness"
    evidence_profile: str = "visible_text_v1"

    def __post_init__(self):
        if not all((self.base_url, self.model, self.revision)):
            raise ValueError("judge endpoint, model and revision must be explicit")
        if min(self.context_window, self.max_new_tokens, self.timeout_seconds) <= 0:
            raise ValueError("judge context, output budget and timeout must be positive")
        if self.criterion_profile not in CRITERION_PROFILES:
            raise ValueError("unknown Granite criterion_profile")
        if self.evidence_profile not in EVIDENCE_PROFILES:
            raise ValueError("unknown Granite evidence_profile")


def guardian_block(*, think=False, criterion_profile="ungroundedness"):
    # The official repository helper includes <guardian>; the tokenizer does not
    # insert it. Keep the final user block separate from the answer being judged.
    return (f"<guardian>{JUDGE_THINK if think else JUDGE_NOTHINK}\n\n"
            f"### Criteria: {CRITERION_PROFILES[criterion_profile]}\n\n### Scoring Schema: {SCORING_SCHEMA}")


def parse_score(text):
    match = re.fullmatch(r"\s*(?:<think>(.*?)</think>\s*)?<score>\s*(yes|no)\s*</score>\s*",
                         text, flags=re.DOTALL | re.IGNORECASE)
    if match is None:
        raise ValueError("expected one native <score>yes/no</score>, optionally preceded by <think>")
    value = match[2].lower()
    return {"native_score": value, "risk_detected": value == "yes",
            "reasoning": match[1].strip() if match[1] is not None else None}


def native_input(record: Mapping, *, think=False, criterion_profile="ungroundedness",
                 evidence_profile="visible_text_v1"):
    if evidence_profile not in EVIDENCE_PROFILES:
        raise ValueError("unknown Granite evidence_profile")
    payload = judge_payload(record)
    by_key = {json.dumps(block["key"], sort_keys=True): block for block in payload["audit_evidence"]}
    current = evidence_profile != "visible_text_v1"
    permitted = None
    if current:
        if "citation_scope" not in payload or not isinstance(payload["citation_scope"], (list, tuple)):
            raise ValueError("current evidence profiles require an explicit citation_scope")
        permitted = set(payload["citation_scope"])
        if any(not isinstance(reference, str) for reference in permitted):
            raise ValueError("citation_scope must contain reference ID strings")
        visible_ids = {reference["reference_id"] for reference in payload["visible_evidence_manifest"]}
        if permitted - visible_ids:
            raise ValueError("allowed evidence has no visible source binding")
    documents, seen = [], set()
    reference_keys = {}
    for reference in payload["visible_evidence_manifest"]:
        if permitted is not None and reference["reference_id"] not in permitted:
            continue
        key = json.dumps(reference["key"], sort_keys=True)
        identity = (key, reference["reference_id"])
        if identity in seen:
            continue
        if key not in by_key:
            raise ValueError("visible evidence is missing its complete original body")
        block = by_key[key]
        if current:
            previous = reference_keys.setdefault(reference["reference_id"], key)
            if previous != key or any(
                other != block for other in payload["audit_evidence"]
                if json.dumps(other["key"], sort_keys=True) == key
            ):
                raise ValueError("current evidence has an ambiguous source binding")
        body = block["text"]
        if evidence_profile == "current_provenance_v1":
            # Match generation's existing provenance, without adding inferred
            # scope, a summary, retrieved targets, or evaluation labels.
            body = _evidence_header(decode(EvidenceBlock, block), reference["reference_id"]) + body + "\n</evidence_body>\n"
        documents.append({"doc_id": reference["reference_id"], "text": body})
        seen.add(identity)
    messages = []
    for message in payload["actual_history"]:
        if message["speaker"] not in {"user", "assistant"}:
            raise ValueError("actual QA history must retain user/assistant roles")
        messages.append({"role": message["speaker"], "content": message["text"]})
    messages.extend([{"role": "user", "content": payload["query"]},
                     {"role": "assistant", "content": payload["answer"]},
                     {"role": "user", "content": guardian_block(think=think, criterion_profile=criterion_profile)}])
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
            "protocol": ("granite_guardian_4_1_ungroundedness_native_v1" if config.criterion_profile == "ungroundedness"
                         else f"granite_guardian_4_1_{config.criterion_profile}"),
            "criterion_profile_version": 1,
            "source_revision": SOURCE_REVISION, "criterion": CRITERION_PROFILES[config.criterion_profile],
            "criterion_origin": "upstream" if config.criterion_profile == "ungroundedness" else "contextkv_byoc",
            "guardian_block": guardian_block(think=config.think, criterion_profile=config.criterion_profile),
            "transport": "native /generate input_ids; no session_id; no second chat template",
            "temperature": 0, "risk_polarity": ("yes=ungroundedness_detected; no=not_detected"
                if config.criterion_profile == "ungroundedness" else "yes=criterion_risk_detected; no=not_detected"),
            "evidence_policy": EVIDENCE_PROFILES[config.evidence_profile] + "; prior assistant turns remain conversation only",
            "quality_gate_eligible": False, "validation_status": "development_diagnostic"}
        if config.evidence_profile != "visible_text_v1":
            self.metadata["protocol"] += "/" + config.evidence_profile

    async def evaluate(self, record):
        result = {"task_id": record["task_id"], "conversation_id": record["conversation_id"],
            "answer": record["answer"], **evaluation_binding(record),
            "source": "offline_granite_guardian", "generation_measurement_kind": record.get("measurement_kind"),
            "judge": self.metadata, "status": "failed", "native_score": None, "risk_detected": None,
            "reference_targets_supplied": False, "quality_noninferiority_established": False}
        before = len(self.client.observations)
        try:
            messages, documents = native_input(record, think=self.config.think,
                criterion_profile=self.config.criterion_profile, evidence_profile=self.config.evidence_profile)
            result.update(messages=messages, documents=documents)
            if self.config.criterion_profile == "citation_binding_v1":
                audit = rule_audit(record)
                result["rule_audit"] = audit
                if not audit["citation_count"]:
                    result.update(status="not_covered", coverage_reason="no_supported_citation_syntax")
                    return result
            ids = list(self.tokenizer.apply_chat_template(messages, documents=documents,
                tokenize=True, return_dict=False, add_generation_prompt=True))
            result.update(judge_input_ids=ids, judge_input_tokens=len(ids))
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
            "failed_task_ids": [r["task_id"] for r in results if r["status"] == "failed"],
            "not_covered_task_ids": [r["task_id"] for r in results if r["status"] == "not_covered"],
            "server_info": info, "protocol": judge.metadata, "quality_accepted": False,
            "interpretation": "binary risk diagnostic under the recorded criterion; no five-dimensional overall score"}
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
