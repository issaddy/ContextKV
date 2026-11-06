"""Offline-only quality evaluation, paired session statistics and audit exports."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict, is_dataclass
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping, Sequence

from controller.schema import OnlineTurn
from bench.metrics import percentile


def paired_cluster_bootstrap(candidate: Mapping[str, Sequence[float]],
                             reference: Mapping[str, Sequence[float]], *, seed: int = 42,
                             samples: int = 5000, confidence: float = .95,
                             epsilon: float = .01, estimand: str = "turn_mean") -> dict:
    """Resample whole conversations, preserving all within-session paired turns.

    Input scores must have been aligned by task_id before grouping by session.
    No missing method/turn is silently dropped. Scores and epsilon use the same
    units, normally [0, 1]; pass percentage points only if scores use that scale.
    """
    if set(candidate) != set(reference) or not candidate:
        raise ValueError("paired methods must cover exactly the same nonempty sessions")
    if samples < 1 or not 0 < confidence < 1 or epsilon < 0:
        raise ValueError("invalid bootstrap configuration")
    if estimand not in {"turn_mean", "session_mean"}:
        raise ValueError("unknown estimand")
    differences = {}
    for sid in sorted(candidate):
        if len(candidate[sid]) != len(reference[sid]) or not candidate[sid]:
            raise ValueError(f"session {sid}: paired turn coverage differs or is empty")
        values = [float(a) - float(b) for a, b in zip(candidate[sid], reference[sid])]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("non-finite quality scores")
        differences[sid] = values
    ids = list(differences)

    def statistic(chosen):
        return mean(mean(differences[sid]) for sid in chosen) if estimand == "session_mean" else mean(v for sid in chosen for v in differences[sid])

    estimate = statistic(ids)
    if len(ids) < 2:
        return {"difference": estimate, "ci_lower": None, "ci_upper": None,
                "decision": "inconclusive", "reason": "fewer than two independent session clusters",
                "clusters": len(ids), "epsilon": epsilon, "estimand": estimand}
    rng = random.Random(seed)
    values = [statistic(rng.choices(ids, k=len(ids))) for _ in range(samples)]
    tail = (1 - confidence) / 2
    low, high = percentile(values, tail), percentile(values, 1 - tail)
    decision = "noninferior" if low > -epsilon else ("inferior" if high < -epsilon else "inconclusive")
    return {"difference": estimate, "ci_lower": low, "ci_upper": high, "decision": decision,
            "clusters": len(ids), "turns": sum(map(len, differences.values())), "epsilon": epsilon,
            "seed": seed, "bootstrap_samples": samples, "confidence": confidence, "estimand": estimand}


def load_qrels(path: str | Path) -> dict[str, dict[str, float]]:
    """Pass the real mtrag-human/retrieval_tasks/<domain>/qrels/dev.tsv path."""
    qrels = defaultdict(dict)
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames or ()):
            raise ValueError("qrels needs query-id, corpus-id and score TSV columns")
        for row in reader:
            qrels[row["query-id"]][row["corpus-id"]] = float(row["score"])
    return dict(qrels)


def load_passage_parents(path: str | Path) -> dict[str, str]:
    """Read an explicit passage-to-document mapping for document-level qrels."""
    parents = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            passage, document = row.get("passage_id"), row.get("document_id")
            if not isinstance(passage, str) or not passage or not isinstance(document, str) or not document:
                raise ValueError("parent corpus needs passage_id and document_id strings")
            if passage in parents:
                raise ValueError(f"duplicate passage ID in parent corpus: {passage}")
            parents[passage] = document
    if not parents:
        raise ValueError("parent corpus must not be empty")
    return parents


def collapse_passage_ranking(passages: Sequence[str], parents: Mapping[str, str]) -> list[str]:
    """Keep the first occurrence of each parent in a passage ranking."""
    missing = set(passages) - parents.keys()
    if missing:
        raise ValueError(f"ranked passage IDs missing from parent corpus: {sorted(missing)}")
    return list(dict.fromkeys(parents[passage] for passage in passages))


def retrieval_quality(retrieved: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, float]], k: int = 5) -> dict:
    """Recall and graded nDCG; unlabelled generation tasks are explicitly missing."""
    if k < 1:
        raise ValueError("k must be positive")
    scores = {}
    missing, no_positive = [], []
    for task_id, passages in retrieved.items():
        if task_id not in qrels:
            missing.append(task_id)
            continue
        labels = qrels[task_id]
        relevant = {pid for pid, score in labels.items() if score > 0}
        if not relevant:
            no_positive.append(task_id)
            continue
        top = list(dict.fromkeys(passages))[:k]
        recall = len(set(top) & relevant) / len(relevant)
        dcg = sum((2 ** max(0, labels.get(pid, 0)) - 1) / math.log2(i + 2) for i, pid in enumerate(top))
        ideal = sum((2 ** score - 1) / math.log2(i + 2) for i, score in enumerate(sorted((v for v in labels.values() if v > 0), reverse=True)[:k]))
        scores[task_id] = {"recall": recall, "ndcg": dcg / ideal}
    return {"k": k, "tasks": scores, "evaluated_tasks": len(scores), "requested_tasks": len(retrieved),
            "missing_qrels_task_ids": sorted(missing), "no_positive_qrels_task_ids": sorted(no_positive),
            "mean_recall": mean(v["recall"] for v in scores.values()) if scores else None,
            "mean_ndcg": mean(v["ndcg"] for v in scores.values()) if scores else None}


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, frozenset, set)):
        return [_jsonable(v) for v in value]
    return value


def official_record(turn: OnlineTurn, answer: str) -> dict:
    """Export the current reading material and preserve raw retrieval separately.

    Preserve the supplied source set. Published MTRAG-UN reference tasks can have
    more contexts than the competition submission checker permits; that check
    belongs to submission validation, not this lossless result export.
    """
    contexts = [{"document_id": c.document_id, "source": c.source, "score": c.score or 0.0,
                 "text": c.text, "title": c.title} for c in (*turn.contexts, *turn.supporting_contexts)]
    messages = [{"speaker": "user" if m.speaker == "user" else "agent", "text": m.text} for m in turn.history]
    messages.append({"speaker": "user", "text": turn.query})
    record = {"task_id": turn.task_id, "conversation_id": turn.conversation_id,
            "Collection": turn.collection, "input": messages,
            "contexts": contexts, "predictions": [{"text": answer}]}
    if turn.retrieval_contexts is not None:
        record["retrieval_contexts"] = [{"document_id": c.document_id, "source": c.source,
            "score": c.score or 0.0, "text": c.text, "title": c.title} for c in turn.retrieval_contexts]
    elif turn.supporting_contexts:
        record["retrieval_contexts"] = contexts[:len(turn.contexts)]
    return record


def evaluation_contexts(turn, manifests, evidence, citation_scope):
    """Project only executed excerpts and the source leaves of final conclusions."""
    blocks = {tuple(sorted(block["key"].items())): block for block in evidence}
    allowed = None if citation_scope is None else set(citation_scope)
    intervals = defaultdict(list)
    for span in manifests:
        if allowed is not None and span["reference_id"] not in allowed:
            continue
        key = tuple(sorted(span["key"].items()))
        block = blocks[key]
        base = block["span"][0] if block["span"] is not None else 0
        start = span.get("source_start")
        end = span.get("source_end")
        intervals[key].append((base if start is None else start,
                              base + len(block["text"]) if end is None else end))
    ranked = {context.document_id: (index, context.score or 0.0)
              for index, context in enumerate((*turn.contexts, *turn.supporting_contexts))}
    contexts = []
    for key, ranges in intervals.items():
        merged = []
        for start, end in sorted(set(ranges)):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        block = blocks[key]
        base = block["span"][0] if block["span"] is not None else 0
        passage = block["key"]["passage_id"]
        contexts.append({"document_id": passage, "source": block["source_uri"],
            "score": ranked.get(passage, (0, 0.0))[1], "title": block["title"],
            "text": "\n\n".join(block["text"][start-base:end-base] for start, end in merged)})
    contexts.sort(key=lambda context: ranked.get(context["document_id"], (len(ranked), 0))[0])
    return contexts


def export_evaluation(turn: OnlineTurn, answer: str, *, official_path: str | Path, audit_path: str | Path,
                      visible_evidence_manifest: Sequence[Any], audit_evidence: Sequence[Any],
                      input_ids: Sequence[int], output_ids: Sequence[int],
                      branch_reason: str | None = None, measurement_kind: str | None = None,
                      truncated: bool | None = None, citation_scope: Sequence[str] | None = None,
                      reading_outcome=None, reference_map=()) -> None:
    if Path(official_path).resolve() == Path(audit_path).resolve():
        raise ValueError("official and full audit outputs must be physically separate")
    full = {"task_id": turn.task_id, "conversation_id": turn.conversation_id, "turn": turn.turn,
            "protocol": turn.protocol, "query": turn.query, "branch_reason": branch_reason, "answer": answer,
            "measurement_kind": measurement_kind, "truncated": truncated,
            "retrieved_evidence": _jsonable(turn.retrieval_contexts if turn.retrieval_contexts is not None else turn.contexts),
            "visible_evidence_manifest": _jsonable(visible_evidence_manifest),
            "audit_evidence": _jsonable(audit_evidence), "input_ids": list(input_ids),
            "output_ids": list(output_ids), "actual_history": _jsonable(turn.history)}
    full["conclusion_source_manifest"] = []
    full["reading_conclusions"] = None
    if reading_outcome is not None and reading_outcome.plan.reading_mode == "llamaindex":
        prior_outputs = []
        for step in reading_outcome.steps:
            if step.task.kind not in {"LLAMAINDEX_COMPACT", "LLAMAINDEX_REFINE"}:
                continue
            observed = step.metrics.get("source_view", {})
            proof = {"step_id": step.step_id, "generation_kind": step.task.kind,
                     "input_hash": observed["input_hash"]}
            full["conclusion_source_manifest"].extend(
                {"key": span["key"], "reference_id": span["reference_id"],
                 "source_start": span["source_start"], "source_end": span["source_end"],
                 "execution_proof": proof}
                for span in _jsonable(observed.get("evidence_spans", ())))
            if step.result is not None:
                prior_outputs.append({**proof, "raw_output": step.result.raw_output})
        full["reading_conclusions"] = {
            "kind": "llamaindex_raw_prior_outputs", "outputs": prior_outputs}
    elif reading_outcome is not None:
        from controller.claim_graph import ClaimGraph
        graph = ClaimGraph(reading_outcome.claims)
        plan = reading_outcome.plan.answer_plan
        conclusions = graph.conclusion_table(plan, reference_map)
        conclusions["open_needs"] = reading_outcome.open_needs
        full["reading_conclusions"] = _jsonable(conclusions)
        by_key = {key: reference for reference, key in reference_map}
        sources = dict.fromkeys(span for point in plan.points for identity in point.claim_ids
                                for span in graph.source_spans(identity))
        full["conclusion_source_manifest"] = [
            {"key": _jsonable(span.source_key), "reference_id": by_key[span.source_key],
             "source_start": span.start, "source_end": span.end} for span in sources]
    if citation_scope is not None:
        full["citation_scope"] = list(citation_scope)
    if turn.query_context is not None:
        full["query_context"] = _jsonable(turn.query_context)
    official = official_record(turn, answer)
    reading = evaluation_contexts(turn,
        (*full["visible_evidence_manifest"], *full["conclusion_source_manifest"]),
        full["audit_evidence"], citation_scope)
    full["evaluation_contexts"] = reading
    if reading != official["contexts"]:
        official.setdefault("retrieval_contexts", [{"document_id": c.document_id, "source": c.source,
            "score": c.score or 0.0, "text": c.text, "title": c.title} for c in turn.contexts])
        official["contexts"] = reading
    for path, record in ((official_path, official), (audit_path, full)):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
