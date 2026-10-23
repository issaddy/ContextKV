"""Offline citation-to-source diagnostics with optional pinned local MiniCheck.

Default export is model-free. Scores are support signals, never ground-truth
labels or an aggregate quality score. Original answers are not rewritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from .judge import CITATION_PATTERN, parse_citations


MINICHECK_REVISION = "b58b9fa69acbd1015ec970fa65dd752413a053d2"
MODELS = {
    "deberta-v3-large": ("lytang/MiniCheck-DeBERTa-v3-Large", "2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb"),
    "flan-t5-large": ("lytang/MiniCheck-Flan-T5-Large", "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"),
}


def claim_spans(answer):
    """Conservative prose spans; preserve offsets and leave code/tables uncovered.

    Newlines/bullets are hard boundaries. Punctuation followed by an uppercase
    word may split prose; boundaries inside inline code are excluded. This is
    an inspectable heuristic, not atomic-claim extraction or decontextualization.
    """
    units, offset, fence = [], 0, None
    for line in answer.splitlines(keepends=True):
        left, right = len(line)-len(line.lstrip()), len(line.rstrip())
        text = line[left:right]
        start = offset+left
        offset += len(line)
        if not text:
            continue
        marker = re.match(r"(`{3,}|~{3,})", text)
        if marker:
            if fence is None:
                fence = marker.group(1)
            elif text.startswith(fence):
                fence = None
            units.append({"start": start, "end": start+len(text), "text": text, "kind": "code"})
            continue
        code_line = re.match(r"(?:from\s+\S+\s+import\b|import\s+|def\s+|class\s+|"
                             r"(?:pip3?|ibmcloud|docker|pod|carthage|curl|npm)\s+)", text)
        heading = text.startswith("#") or re.fullmatch(r"(?:\d+[.)]\s+)?\*\*[^*]+\*\*:?[ ]*", text)
        kind = ("code" if fence or code_line else "table" if "|" in text and text.count("|") >= 2
                else "heading" if heading else "prose")
        if kind != "prose":
            units.append({"start": start, "end": start+len(text), "text": text, "kind": kind})
            continue
        code_ranges = [(m.start(), m.end()) for m in re.finditer(r"`[^`]*`", text)]
        boundaries = [0]+[m.end() for m in re.finditer(r"[.!?]\s+(?=[A-Z])", text)
                          if not any(a <= m.start() < b for a, b in code_ranges)]+[len(text)]
        for begin, end in zip(boundaries, boundaries[1:]):
            part = text[begin:end]
            trim = len(part)-len(part.lstrip())
            part = part.strip()
            if not part:
                continue
            begin += trim
            remaining = CITATION_PATTERN.sub("", part)
            remaining = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", remaining)
            remaining = re.sub(r"(?i)\b(?:evidence|source|id)\b", "", remaining)
            citation_only = bool(parse_citations(part)) and not re.search(r"[A-Za-z0-9]", remaining)
            units.append({"start": start+begin, "end": start+begin+len(part), "text": part,
                "kind": "ambiguous_citation_binding" if citation_only else "prose",
                "contains_inline_code": "`" in part})
    return units


def grounding_record(record, scorer=None, *, threshold=.5):
    bodies = {json.dumps(block["key"], sort_keys=True): block for block in record["audit_evidence"]}
    blocks, missing = {}, []
    for span in record["visible_evidence_manifest"]:
        block = bodies.get(json.dumps(span["key"], sort_keys=True))
        if block is None:
            missing.append(span["reference_id"])
        else:
            blocks[span["reference_id"]] = block
    claims = []
    for index, unit in enumerate(claim_spans(record["answer"])):
        citations = sorted(parse_citations(unit["text"]))
        claim = {"claim_index": index, **unit, "citations": citations,
            "unknown_citations": sorted(set(citations)-set(blocks)),
            "missing_citation": not citations, "pairs": [], "finding": "not_covered"}
        if unit["kind"] != "prose":
            claims.append(claim)
            continue
        # Preserve every visible block, including old/revisited material. Score
        # cited blocks first; alternative sources diagnose misbinding separately.
        refs = sorted(blocks, key=lambda ref: (ref not in citations, ref))
        # Reference IDs are bindings, not factual assertions. Keep their exact
        # offsets above, but exclude these labels from the entailment input.
        semantic_claim = CITATION_PATTERN.sub("", unit["text"]).strip()
        claim["semantic_claim"] = semantic_claim
        pairs = [{"reference_id": ref, "cited": ref in citations, "key": blocks[ref]["key"],
            "source_uri": blocks[ref]["source_uri"], "document": blocks[ref]["text"],
            "claim": semantic_claim, "status": "dry_run", "probability": None,
            "support_signal": None} for ref in refs]
        if scorer is not None and pairs:
            outputs = scorer.score_pairs([(p["document"], p["claim"]) for p in pairs])
            if len(outputs) != len(pairs):
                raise ValueError("scorer results must align with exported source pairs")
            for pair, output in zip(pairs, outputs):
                pair.update(output)
                pair["support_signal"] = (pair["probability"] >= threshold
                    if pair["status"] == "scored" else None)
        claim["pairs"] = pairs
        usable = [pair for pair in pairs if pair["status"] == "scored"]
        supporting = [pair["reference_id"] for pair in usable if pair["support_signal"]]
        cited_supported = set(supporting) & set(citations)
        incomplete = bool(missing or len(usable) != len(pairs))
        if scorer is None:
            finding = "dry_run"
        elif not pairs or incomplete:
            finding = "incomplete_model_coverage"
        elif not citations:
            finding = "uncited_support_signal" if supporting else "uncited_no_support_detected"
        elif cited_supported:
            finding = "cited_support_signal" if set(citations) <= cited_supported else "some_citations_lack_support_signal"
        elif supporting:
            finding = "supported_elsewhere_citation_mismatch"
        else:
            finding = "no_support_detected"
        claim.update(finding=finding, supporting_reference_ids=supporting,
                     assessment_scope="prose support signal; inline code behavior is not verified")
        claims.append(claim)
    return {"task_id": record["task_id"], "conversation_id": record.get("conversation_id"),
        "protocol": record.get("protocol"), "query": record.get("query"), "answer": record["answer"],
        "source_truncated": record.get("truncated"), "missing_visible_bodies": sorted(set(missing)),
        "visible_reference_ids": sorted(blocks), "claims": claims,
        "threshold": threshold, "threshold_calibrated": False,
        "segmentation": "conservative_offset_prose_v1; code/table/ambiguous bindings are not covered",
        "claim_preprocessing": "citation_markers_removed_for_entailment; original_text_and_binding_preserved",
        "semantic_quality_score": None, "answer_modified": False}


class MiniCheckScorer:
    """Small Apache-2.0 loader adaptation; scoring/chunking remain upstream.

    Upstream MiniCheck inference.py initialization hardcodes main/auto. We only
    replace that initializer to load an explicit local snapshot on one device.
    Official MiniCheck.score and Inferencer.fact_check/inference are reused.
    """
    def __init__(self, model, model_path, minicheck_root, *, device="cpu", max_length=2048,
                 batch_size=8, chunk_size=400):
        root = Path(minicheck_root).resolve()
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if revision != MINICHECK_REVISION:
            raise ValueError(f"MiniCheck source must be pinned to {MINICHECK_REVISION}")
        sys.path.insert(0, str(root))
        from minicheck.minicheck import MiniCheck
        from minicheck.inference import Inferencer
        import inspect
        import nltk
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
        if not Path(inspect.getfile(MiniCheck)).resolve().is_relative_to(root):
            raise ValueError("an already imported MiniCheck package differs from the pinned source root")
        # Fail with NLTK's resource description; never download inside scoring.
        nltk.sent_tokenize("Offline resource check.")
        snapshot = str(Path(model_path).resolve())

        class PinnedInferencer(Inferencer):
            def __init__(self):
                self.model_name, self.max_model_len, self.batch_size = model, max_length, batch_size
                self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
                loader = AutoModelForSeq2SeqLM if model == "flan-t5-large" else AutoModelForSequenceClassification
                self.model = loader.from_pretrained(snapshot, local_files_only=True).to(device).eval()
                self.max_output_length = 256
                self.softmax = torch.nn.Softmax(dim=-1)
                self.tokenization_audit = []

            def batch_tokenize(self, docs, claims):
                for doc, claim in zip(docs, claims):
                    value = self.tokenizer.eos_token.join((doc, claim))
                    if model == "flan-t5-large":
                        value = "predict: "+value
                    count = len(self.tokenizer(value, truncation=False)["input_ids"])
                    self.tokenization_audit.append({"document_chunk": doc, "input_tokens": count,
                        "max_length": max_length, "truncated": count > max_length})
                return super().batch_tokenize(docs, claims)

        class PinnedMiniCheck(MiniCheck):
            def __init__(self):
                self.model = PinnedInferencer()

        self.scorer, self.chunk_size = PinnedMiniCheck(), chunk_size
        self.provenance = {"implementation": "official MiniCheck scoring with local pinned loader adaptation",
            "repository": "https://github.com/Liyan06/MiniCheck", "repository_revision": revision,
            "code_license": "Apache-2.0", "model_id": MODELS[model][0], "model_revision": MODELS[model][1],
            "model_license": "MIT", "model_path": snapshot,
            "model_revision_provenance": "operator_supplied_local_snapshot; weights are not rehashed",
            "device": str(self.scorer.model.model.device), "dtype": str(self.scorer.model.model.dtype),
            "max_length": max_length, "chunk_size": chunk_size, "batch_size": batch_size,
            "model_downloads": False, "model_source": "local_pretrained_grounding_classifier"}

    def score_pairs(self, pairs):
        results = []
        for doc, claim in pairs:
            self.scorer.model.tokenization_audit = []
            started = time.perf_counter()
            labels, probabilities, chunks, chunk_probabilities = self.scorer.score(
                docs=[doc], claims=[claim], chunk_size=self.chunk_size)
            audit = self.scorer.model.tokenization_audit
            status = "model_truncated" if any(item["truncated"] for item in audit) else "scored"
            if not audit:
                status = "not_covered"
            results.append({"status": status, "probability": float(probabilities[0]),
                "upstream_label": int(labels[0]), "used_chunks": chunks[0],
                "chunk_probabilities": [float(value) for value in chunk_probabilities[0]],
                "tokenization": audit, "elapsed_ms": (time.perf_counter()-started)*1000})
        return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, help="full audit JSONL, never official top-k export")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=tuple(MODELS), help="explicitly enable local classifier; omitted means dry-run")
    parser.add_argument("--model-path")
    parser.add_argument("--minicheck-root", default="vendor/minicheck")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="official DeBERTa/Flan-T5 default; actual per-chunk truncation is recorded")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=.5)
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1 or min(args.max_length, args.chunk_size, args.batch_size) < 1:
        parser.error("threshold must be in [0,1] and model sizes positive")
    if args.model and not args.model_path:
        parser.error("--model requires an explicitly prepared --model-path snapshot")
    output = Path(args.output)
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError("grounding output exists; select a new file")
    scorer = (MiniCheckScorer(args.model, args.model_path, args.minicheck_root,
              device=args.device, max_length=args.max_length, batch_size=args.batch_size, chunk_size=args.chunk_size)
              if args.model else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"operation": "dry_run" if scorer is None else "local_model_diagnostic",
        "source_audit": str(Path(args.audit).resolve()), "arguments": vars(args),
        "scorer": scorer.provenance if scorer else None, "semantic_quality_score": None,
        "paid_judge_calls": 0, "answer_modified": False, "status": "running"}
    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    try:
        with output.open("x", encoding="utf-8") as target, Path(args.audit).open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    record = grounding_record(json.loads(line), scorer, threshold=args.threshold)
                    target.write(json.dumps(record, ensure_ascii=False)+"\n")
                    target.flush()
    except Exception:
        metadata["status"] = "failed_partial_output_preserved"
        raise
    else:
        metadata["status"] = "complete"
    finally:
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
