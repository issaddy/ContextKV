"""Optional offline claim ledger, not a five-dimension quality gate.

Method adaptation inspired by RefChecker and context-preserving claim extraction;
not a complete reproduction. Quote matching proves provenance, not entailment.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from importlib.resources import files
import json
import math
from pathlib import Path
import re
from typing import Mapping

from bench.grounding import claim_spans
from bench.judge import evaluation_binding, judge_payload, parse_citations, rule_audit
from bench.native_judge import NativeJudgeClient, NativeJudgeConfig
from bench.official import load_jsonl

KINDS = {"assertion", "abstention", "recommendation", "hypothetical", "ambiguous"}
VERDICTS = {"entailed", "contradicted", "insufficient", "ambiguous", "not_verifiable"}
RELATIONS = {"supports", "opposes", "qualifies"}


def _object(text):
    fenced = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text, re.DOTALL)
    value = json.loads(fenced[1] if fenced else text)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _positions(text, quote):
    return [match.start() for match in re.finditer("(?="+re.escape(quote)+")", text)] if quote else []


def _sources(record):
    # Resolve existing structured identities once at the offline input boundary.
    bodies = {json.dumps(b["key"], sort_keys=True): b for b in record["audit_evidence"]}
    return [{"reference_id": s["reference_id"], "text": bodies[json.dumps(s["key"], sort_keys=True)]["text"],
             "key": s["key"]} for s in record["visible_evidence_manifest"]]


def _bind_claim(claim, units, seen_ids):
    if not isinstance(claim, dict):
        raise ValueError("claim entry must be an object")
    for key in ("claim_id", "unit_id", "answer_quote", "statement", "kind"):
        if not isinstance(claim.get(key), str) or not claim[key].strip():
            raise ValueError(f"claim requires nonempty {key}")
    if claim["claim_id"] in seen_ids:
        raise ValueError("duplicate claim_id")
    seen_ids.add(claim["claim_id"])
    if claim["kind"] not in KINDS:
        raise ValueError("unknown claim kind")
    for key in ("object", "relation", "unit_raw", "comparator", "scope", "modality"):
        if key not in claim or (claim[key] is not None and not isinstance(claim[key], str)):
            raise ValueError(f"claim requires explicit string-or-null {key}")
    value = claim.get("value_raw")
    if ("value_raw" not in claim or (value is not None and type(value) not in {str, int, float})
            or (type(value) is float and not math.isfinite(value))):
        raise ValueError("value_raw must be an explicit string, finite number, or null")
    if claim["modality"] not in {None, "asserted", "conditional", "recommended", "hypothetical", "uncertain"}:
        raise ValueError("unknown claim modality")
    if not isinstance(claim.get("conditions"), list) or any(not isinstance(x, str) for x in claim["conditions"]):
        raise ValueError("conditions must be an explicit list of strings")
    if "negation" not in claim or (claim["negation"] is not None and type(claim["negation"]) is not bool):
        raise ValueError("negation must be an explicit boolean or null")
    unit = units.get(claim["unit_id"])
    if unit is None:
        raise ValueError("unknown answer unit")
    if unit["kind"] in {"code", "table", "ambiguous_citation_binding"}:
        raise ValueError("unit requires separate structural or citation-binding review")
    locations = _positions(unit["text"], claim["answer_quote"])
    if len(locations) != 1:
        raise ValueError("answer_quote must match exactly one span in the declared unit")
    citations = sorted(parse_citations(unit["text"]))
    declared = claim.get("citation_ids")
    if not isinstance(declared, list) or any(not isinstance(x, str) for x in declared):
        raise ValueError("citation_ids must be a list of original IDs")
    start = unit["start"] + locations[0]
    return {"unit_id": unit["unit_id"], "start": start, "end": start+len(claim["answer_quote"]),
            "text": claim["answer_quote"], "unit_text": unit["text"],
            "citation_ids": citations, "extractor_citation_ids": declared,
            "extra_extractor_citation_ids": sorted(set(declared) - set(citations)),
            "citation_scope": "original_unit_group_not_proven_claim_attachment",
            "meaning": "exact location only; extraction faithfulness still model-assessed"}


def _bind_check(value, claim_id, sources):
    if value.get("claim_id") != claim_id or value.get("verdict") not in VERDICTS:
        raise ValueError("checker requires matching claim_id and a known verdict")
    faithful = value.get("extraction_faithful")
    if faithful is not None and type(faithful) is not bool:
        raise ValueError("extraction_faithful must be boolean or null")
    if "extraction_faithful" not in value:
        raise ValueError("checker must assess extraction_faithful explicitly")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError("checker requires a reason")
    if not isinstance(value.get("failed_slots"), list) or any(not isinstance(x, str) for x in value["failed_slots"]):
        raise ValueError("failed_slots must be a list of strings")
    answerable = value.get("source_answers_current_question")
    if answerable is not None and type(answerable) is not bool:
        raise ValueError("source_answers_current_question must be boolean or null")
    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("checker requires evidence list")
    by_id = {source["reference_id"]: source for source in sources}
    bound = []
    for item in evidence:
        if not isinstance(item, dict) or item.get("reference_id") not in by_id:
            raise ValueError("checker cited a source outside this check's source subset")
        if not isinstance(item.get("quote"), str) or not item["quote"]:
            raise ValueError("evidence requires a nonempty exact source quote")
        if item.get("relation") not in RELATIONS:
            raise ValueError("unknown evidence relation")
        source = by_id[item["reference_id"]]
        positions = _positions(source["text"], item["quote"])
        if not positions:
            raise ValueError("evidence quote is not an exact substring of its source")
        bound.append({**item, "source_spans": [{"start": p, "end": p+len(item["quote"])} for p in positions],
                      "key": source["key"], "meaning": "location verified; support/opposition remains a model judgment"})
    needed_relation = {"entailed": "supports", "contradicted": "opposes"}.get(value["verdict"])
    if needed_relation and not any(e["relation"] == needed_relation for e in bound):
        raise ValueError(f"{value['verdict']} requires a {needed_relation} source quote")
    return bound


class AtomicJudgeClient:
    """Reuse native_judge's raw transport, not its native scoring templates."""
    def __init__(self, config: NativeJudgeConfig, *, transport=None, token_counter=None):
        self.transport = NativeJudgeClient(config, transport=transport, token_counter=token_counter)
        self.protocol = json.loads(files("bench").joinpath("assets/atomic_judge/protocol.json").read_text())
        used = ("provider", "model", "revision", "base_url", "temperature", "max_tokens", "timeout_seconds",
                "api_key_env", "base_model", "base_family", "context_window")
        configuration = asdict(config)
        self.metadata = {k: configuration[k] for k in used}
        self.metadata.update(protocol=self.protocol["version"], protocol_manifest=self.protocol,
            model_revision_provenance="operator_declared",
            context_budget="exact_chat_tokens_plus_max_tokens" if token_counter else "not_checked")

    async def _json_call(self, stage, payload):
        response = await self.transport.complete([
            {"role": "system", "content": self.protocol[stage+"_system"]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        if response["status"] == "completed":
            try:
                response["parsed"] = _object(response["raw_text"])
            except (ValueError, TypeError) as exc:
                response.update(status="failed", error=str(exc))
        return response

    async def _check(self, record, claim, binding, sources):
        response = await self._json_call("check", {
            "query": record["query"], "actual_history": record["actual_history"],
            "claim": {**claim, "citation_ids": binding["citation_ids"]}, "original_answer": record["answer"],
            "answer_binding": binding, "sources": sources})
        if response["status"] == "completed":
            try:
                value = response["parsed"]
                response["bound_evidence"] = _bind_check(value, claim["claim_id"], sources)
                # An exact quote is never converted into an entailment decision.
                resolved = (value["extraction_faithful"] is True and
                            (claim["kind"] != "abstention" or type(value.get("source_answers_current_question")) is bool))
                response.update(status="assessed" if resolved else "unresolved",
                                verdict=value["verdict"])
            except (ValueError, TypeError, KeyError) as exc:
                response.update(status="failed", error=str(exc))
        return response

    async def evaluate(self, record: Mapping) -> dict:
        payload = judge_payload(record)  # shared audit whitelist; never join gold labels
        rules = rule_audit(record)
        result = {"task_id": record["task_id"], "conversation_id": record["conversation_id"],
            "query": record["query"], "answer": record["answer"], **evaluation_binding(record),
            "source": "atomic_judge_diagnostic", "diagnostic_only": True,
            "formal_quality_eligible": False, "judge": self.metadata, "rule_audit": rules,
            "generation_measurement_kind": record.get("measurement_kind"), "claims": [],
            "status": "failed", "extraction": None}
        if "missing_visible_evidence_body" in rules["flags"]:
            result["error"] = "missing full visible evidence body"
            return result
        units = [{"unit_id": f"u{i+1}", **unit} for i, unit in enumerate(claim_spans(payload["answer"]))]
        unit_map = {unit["unit_id"]: unit for unit in units}
        result["answer_units"] = units
        if not units:
            result.update(status="unassessable", error="empty answer; no factuality score inferred")
            return result
        sources = _sources(record)
        by_source = {source["reference_id"]: source for source in sources}
        extraction = await self._json_call("extract", {
            "query": payload["query"], "actual_history": payload["actual_history"], "answer_units": units})
        result["extraction"] = extraction
        if extraction["status"] == "failed":
            return result
        extracted = extraction["parsed"]
        if not isinstance(extracted.get("claims"), list) or not isinstance(extracted.get("uncovered_units"), list):
            extraction.update(status="failed", error="extraction needs claims and uncovered_units lists")
            return result
        seen_ids, covered, deduplicated = set(), set(), []
        for index, claim in enumerate(extracted["claims"]):
            entry = {"entry_index": index, "extracted": claim, "status": "failed"}
            result["claims"].append(entry)
            try:
                binding = _bind_claim(claim, unit_map, seen_ids)
            except (ValueError, TypeError) as exc:
                entry["error"] = str(exc)
                continue
            entry["answer_binding"] = binding
            entry["citation_declaration"] = {
                "status": "extra_ids_not_bound_to_unit" if binding["extra_extractor_citation_ids"] else "no_extra_ids",
                "declared_ids": binding["extractor_citation_ids"],
                "original_unit_ids": binding["citation_ids"],
                "extra_ids": binding["extra_extractor_citation_ids"],
                "meaning": "extractor declarations do not select sources or establish claim attachment"}
            identity = {k: v for k, v in claim.items() if k != "claim_id"}
            duplicate = next((previous for value, previous in deduplicated if value == identity), None)
            if duplicate is not None:
                entry.update(status="duplicate", duplicate_of=duplicate)
                continue
            deduplicated.append((identity, index))
            covered.add(binding["unit_id"])
            visible = await self._check(record, claim, binding, sources)
            entry["visible_check"] = visible
            citation_ids = binding["citation_ids"]
            unknown = sorted(set(citation_ids) - set(by_source))
            entry["unknown_citations"] = unknown
            if not citation_ids:
                entry["citation_status"] = "missing" if not parse_citations(record["answer"]) else "not_bound_to_unit"
                entry["cited_check"] = None
            else:
                cited_sources = [s for s in sources if s["reference_id"] in citation_ids]
                if not cited_sources:
                    entry["cited_check"] = None
                elif len(cited_sources) == len(sources):
                    entry["cited_check"] = {"reused_visible_check": True, **visible}
                else:
                    entry["cited_check"] = await self._check(record, claim, binding, cited_sources)
                entry["citation_status"] = "unknown_id" if unknown else "assessed_source_group"
            checks = [visible] + ([entry["cited_check"]] if entry["cited_check"] else [])
            entry["status"] = "assessed" if all(c["status"] == "assessed" for c in checks) else "unresolved"
        declared, coverage_errors = {}, []
        for item in extracted["uncovered_units"]:
            if (not isinstance(item, dict) or not isinstance(item.get("unit_id"), str)
                    or item["unit_id"] not in unit_map or not isinstance(item.get("reason"), str)):
                coverage_errors.append({"entry": item, "error": "uncovered entry requires an original unit_id and reason"})
            else:
                declared[item["unit_id"]] = item["reason"]
        unaccounted = sorted(set(unit_map) - covered - set(declared))
        assessed = [e for e in result["claims"] if e["status"] == "assessed"]
        visible_assessed = [e for e in result["claims"] if e.get("visible_check", {}).get("status") == "assessed"]
        result["coverage"] = {"kind": "unit_accounting_not_semantic_completeness", "total_units": len(units),
            "units_with_bound_claims": sorted(covered), "declared_uncovered": declared,
            "unaccounted_units": unaccounted, "errors": coverage_errors,
            "structural_review_units": [u["unit_id"] for u in units if u["kind"] in {"code", "table"}],
            "unbound_citation_units": [u["unit_id"] for u in units if u["kind"] == "ambiguous_citation_binding"],
            "citation_declaration_errors": [e["entry_index"] for e in result["claims"]
                if e.get("citation_declaration", {}).get("extra_ids")],
            "meaning": "A located claim does not prove that all qualifiers or facts were extracted."}
        result["summary"] = {"unique_bound_claims": len(deduplicated), "assessed_claims": len(assessed),
            "visible_assessed_claims": len(visible_assessed),
            "visible_model_verdict_counts": {v: sum(e["visible_check"].get("verdict") == v for e in visible_assessed) for v in sorted(VERDICTS)},
            "unknown_or_failed_claims": sum(e["status"] in {"failed", "unresolved"} for e in result["claims"]),
            "support_fraction": None, "quality_score": None}
        uncovered_review = any(reason not in {"nonverifiable", "heading"} for reason in declared.values())
        incomplete = (unaccounted or coverage_errors or uncovered_review or result["coverage"]["structural_review_units"]
                      or result["coverage"]["unbound_citation_units"]
                      or result["coverage"]["citation_declaration_errors"]
                      or any(e["status"] in {"failed", "unresolved"} for e in result["claims"]))
        result["status"] = "partial" if incomplete else "complete"
        if not result["claims"]:
            result["status"] = "failed" if coverage_errors else "partial" if unaccounted else "unassessable"
        return result

    async def aclose(self):
        await self.transport.aclose()


async def evaluate_file(audit_path, config, output_path, *, transport=None, token_counter=None):
    rows = load_jsonl(audit_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        client = AtomicJudgeClient(config, transport=transport, token_counter=token_counter)
        try:
            for row in rows:
                result = await client.evaluate(row)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
        finally:
            await client.aclose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--config", required=True, help="native judge config; transport/model/budget fields are reused")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-path", help="fixed local tokenizer, never downloaded")
    args = parser.parse_args(argv)
    config = NativeJudgeConfig(**json.loads(Path(args.config).read_text()))
    counter = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, revision=config.revision,
                                                   local_files_only=True, trust_remote_code=False)
        def counter(messages):
            return len(tokenizer.apply_chat_template(messages, tokenize=True,
                                                       add_generation_prompt=True, return_dict=False))
    asyncio.run(evaluate_file(args.audit, config, args.output, token_counter=counter))


if __name__ == "__main__":
    main()
