"""Pinned local LettuceDetect span diagnostics, not a quality acceptance gate.

Small MIT adaptation of KRLabsOrg/LettuceDetect; see assets/lettuce_judge/NOTICE.md.
The native QA pair is scored whole, without truncation or max-over-chunks.
"""
from __future__ import annotations

import argparse
from importlib.resources import files
import json
import math
from pathlib import Path
from string import Template
import time

from bench.judge import CITATION_PATTERN, evaluation_binding, judge_payload, rule_audit

ASSETS = files("bench").joinpath("assets/lettuce_judge")
PROTOCOL = json.loads(ASSETS.joinpath("protocol.json").read_text())
QA_TEMPLATE = ASSETS.joinpath("qa_prompt_en.txt").read_text()


def format_prompt(context: list[str], query: str) -> str:
    """Pinned upstream English PromptUtils.format_context QA behavior."""
    passages = "\n".join(f"passage {i+1}: {text}" for i, text in enumerate(context))
    return Template(QA_TEMPLATE).substitute(question=query, num_passages=len(context), context=passages)


def _decode_tokens(answer, input_ids, offsets, sequence_ids, special_mask, probabilities):
    if len(probabilities) != len(input_ids):
        raise ValueError("classifier probabilities do not align with the complete input")
    tokens, spans, current = [], [], None
    citations = [(m.start(), m.end()) for m in CITATION_PATTERN.finditer(answer)]
    for index, (offset, sequence, special) in enumerate(zip(offsets, sequence_ids, special_mask)):
        start, end = offset
        if sequence != 1 or special or start == end:
            continue
        if not 0 <= start < end <= len(answer):
            raise ValueError("answer-side tokenizer offset is outside the original answer")
        scores = probabilities[index]
        if (len(scores) != 2 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores)):
            raise ValueError("classifier must return two finite class probabilities")
        label = int(scores[1] > scores[0])  # upstream argmax: first class wins ties
        overlap = any(start < right and end > left for left, right in citations)
        tokens.append({"input_token_index": index, "token_id": input_ids[index],
            "start": start, "end": end, "text": answer[start:end], "label_id": label,
            "unsupported_probability": scores[1], "class_probabilities": list(scores),
            "overlaps_citation_marker": overlap})
        if label == 1:
            if current is None:
                current = {"start": start, "end": end, "confidence": scores[1], "token_count": 1}
            else:
                current.update(end=max(current["end"], end),
                               confidence=max(current["confidence"], scores[1]),
                               token_count=current["token_count"]+1)
        elif current is not None:
            spans.append(current)
            current = None
    if current is not None:
        spans.append(current)
    for span in spans:
        span["text"] = answer[span["start"]:span["end"]]
        span["citation_marker_intersections"] = [
            {"start": max(span["start"], a), "end": min(span["end"], b)}
            for a, b in citations if span["start"] < b and span["end"] > a]
    covered = {i for token in tokens for i in range(token["start"], token["end"])}
    uncovered = [i for i, char in enumerate(answer) if not char.isspace() and i not in covered]
    return {"tokens": tokens, "unsupported_spans": spans,
            "coverage": {"answer_tokens": len(tokens), "unscored_nonwhitespace_character_positions": uncovered,
                         "meaning": "token coverage only, not semantic correctness or completeness"}}


class LettuceScorer:
    """Tokenizer plus one forward callback; tests can inject a CPU-only fake."""
    def __init__(self, tokenizer, predict_probabilities, *, max_length=8192, provenance=None):
        if not 0 < max_length <= PROTOCOL["context_limit_tokens"]:
            raise ValueError("max_length must be in 1..8192 for this fixed model")
        self.tokenizer, self.predict_probabilities = tokenizer, predict_probabilities
        self.max_length, self.provenance = max_length, provenance or {"model_execution": "injected_test_backend"}

    @classmethod
    def from_local_snapshot(cls, model_path, *, device="cpu", max_length=8192):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        if not 0 < max_length <= PROTOCOL["context_limit_tokens"]:
            raise ValueError("max_length must be in 1..8192 for this fixed model")
        snapshot = str(Path(model_path).resolve())
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        model = AutoModelForTokenClassification.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False).to(device).eval()
        labels = {int(k): v for k,v in model.config.id2label.items()}
        if labels != {0: "supported", 1: "hallucinated"}:
            raise ValueError("local model label direction differs from the fixed LettuceDetect checkpoint")
        if max_length > model.config.max_position_embeddings:
            raise ValueError("max_length exceeds the loaded model context")

        def predict_probabilities(encoding):
            tensors = {key: torch.tensor([encoding[key]], device=device) for key in ("input_ids", "attention_mask")}
            with torch.inference_mode():
                logits = model(**tensors).logits
                return torch.softmax(logits, dim=-1)[0].cpu().tolist()

        provenance = {"model_path": snapshot, "model_execution": "local_pretrained_classifier",
            "model_revision_provenance": "operator_supplied_local_snapshot; weights not rehashed",
            "device": str(model.device), "dtype": str(model.dtype),
            "architecture": model.config.architectures, "model_context_tokens": model.config.max_position_embeddings,
            "tokenizer_model_max_length": tokenizer.model_max_length, "model_downloads": False}
        return cls(tokenizer, predict_probabilities, max_length=max_length, provenance=provenance)

    def score(self, prompt, answer):
        encoded = self.tokenizer(prompt, answer, add_special_tokens=True, truncation=False,
                                 padding=False, return_offsets_mapping=True, return_special_tokens_mask=True)
        input_ids = encoded["input_ids"]
        offsets, sequences, special = encoded["offset_mapping"], encoded.sequence_ids(), encoded["special_tokens_mask"]
        if not len(input_ids) == len(offsets) == len(sequences) == len(special):
            raise ValueError("tokenizer pair fields do not align")
        result = {"input_tokens": len(input_ids), "max_length": self.max_length,
                  "model_inference_attempted": False, "model_inference_performed": False,
                  "truncated": False, "chunk_count": 0}
        if len(input_ids) > self.max_length:
            return {**result, "status": "not_covered", "reason": "complete_input_exceeds_context"}
        if not any(s == 1 and not m and a < b for s,m,(a,b) in zip(sequences, special, offsets)):
            return {**result, "status": "not_covered", "reason": "no_answer_tokens"}
        started = time.perf_counter()
        result.update(model_inference_attempted=True, chunk_count=1)
        try:
            probabilities = self.predict_probabilities(encoded)
            result["model_inference_performed"] = True
            decoded = _decode_tokens(answer, input_ids, offsets, sequences, special, probabilities)
        except Exception as exc:
            return {**result, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_ms": (time.perf_counter()-started)*1000}
        status = "partial" if decoded["coverage"]["unscored_nonwhitespace_character_positions"] else "scored"
        return {**result, **decoded, "status": status, "model_inference_performed": True,
                "chunk_count": 1, "elapsed_ms": (time.perf_counter()-started)*1000,
                "unsupported_detected": bool(decoded["unsupported_spans"])}


def evaluate_audit(record, scorer=None):
    payload = judge_payload(record)  # source audit whitelist; references/labels never requested
    result = {"task_id": record["task_id"], "conversation_id": record["conversation_id"],
        "query": payload["query"], "answer": payload["answer"], **evaluation_binding(record),
        "source": "lettuce_span_diagnostic", "diagnostic_only": True, "formal_quality_eligible": False,
        "generation_measurement_kind": record.get("measurement_kind"), "source_truncated": record.get("truncated"),
        "rule_audit": rule_audit(record), "status": "not_covered",
        "model_inference_attempted": False, "model_inference_performed": False,
        "judge": {"model": PROTOCOL["model"], "revision": PROTOCOL["model_revision"],
                  "base_family": PROTOCOL["base_family"], "protocol": PROTOCOL["version"],
                  "max_input_tokens": scorer.max_length if scorer else None,
                  "protocol_manifest": {**PROTOCOL, "qa_template": QA_TEMPLATE},
                  "runtime": scorer.provenance if scorer else {"model_execution": "not_loaded"}},
        "history_policy": PROTOCOL["history_policy"], "history_turns_not_submitted": len(payload["actual_history"]),
        "semantic_quality_score": None, "citation_correctness_score": None}
    bodies = {json.dumps(b["key"], sort_keys=True): b for b in payload["audit_evidence"]}
    sources = []
    for visible in payload["visible_evidence_manifest"]:
        body = bodies.get(json.dumps(visible["key"], sort_keys=True))
        if body is None or not body["text"].strip():
            return {**result, "reason": "missing_or_empty_visible_evidence_body"}
        sources.append({"reference_id": visible["reference_id"], "key": visible["key"], "text": body["text"]})
    if not payload["answer"].strip() or not sources:
        return {**result, "reason": "empty_answer" if not payload["answer"].strip() else "no_visible_sources"}
    prompt = format_prompt([s["text"] for s in sources], payload["query"])
    result.update(model_input={"prompt": prompt, "answer": payload["answer"]}, visible_sources=sources)
    if scorer is None:
        return {**result, "status": "dry_run", "reason": "model_not_loaded"}
    try:
        result.update(scorer.score(prompt, payload["answer"]))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return result


def evaluate_file(audit_path, output_path, scorer=None):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(audit_path).open() as source, output.open("x") as stream:
        for line in source:
            if line.strip():
                stream.write(json.dumps(evaluate_audit(json.loads(line), scorer), ensure_ascii=False)+"\n")
                stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", help="fixed local snapshot; omitted means model-free export")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args(argv)
    scorer = LettuceScorer.from_local_snapshot(args.model_path, device=args.device,
                                               max_length=args.max_length) if args.model_path else None
    evaluate_file(args.audit, args.output, scorer)


if __name__ == "__main__":
    main()
