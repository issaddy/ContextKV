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

from adapters.mtrag import OnlineTurn
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
    """Official export uses current retrieval; full visible evidence is separate.

    Preserve the supplied source set. Published MTRAG-UN reference tasks can have
    more contexts than the competition submission checker permits; that check
    belongs to submission validation, not this lossless result export.
    """
    contexts = [{"document_id": c.document_id, "source": c.source, "score": c.score or 0.0,
                 "text": c.text, "title": c.title} for c in turn.contexts]
    messages = [{"speaker": "user" if m.speaker == "user" else "agent", "text": m.text} for m in turn.history]
    messages.append({"speaker": "user", "text": turn.query})
    return {"task_id": turn.task_id, "conversation_id": turn.conversation_id,
            "Collection": turn.collection, "input": messages,
            "contexts": contexts, "predictions": [{"text": answer}]}


def export_evaluation(turn: OnlineTurn, answer: str, *, official_path: str | Path, audit_path: str | Path,
                      visible_evidence_manifest: Sequence[Any], audit_evidence: Sequence[Any],
                      input_ids: Sequence[int], output_ids: Sequence[int],
                      branch_reason: str | None = None, measurement_kind: str | None = None,
                      truncated: bool | None = None, citation_scope: Sequence[str] | None = None) -> None:
    if Path(official_path).resolve() == Path(audit_path).resolve():
        raise ValueError("official and full audit outputs must be physically separate")
    full = {"task_id": turn.task_id, "conversation_id": turn.conversation_id, "turn": turn.turn,
            "protocol": turn.protocol, "query": turn.query, "branch_reason": branch_reason, "answer": answer,
            "measurement_kind": measurement_kind, "truncated": truncated,
            "retrieved_evidence": _jsonable(turn.contexts),
            "visible_evidence_manifest": _jsonable(visible_evidence_manifest),
            "audit_evidence": _jsonable(audit_evidence), "input_ids": list(input_ids),
            "output_ids": list(output_ids), "actual_history": _jsonable(turn.history)}
    if citation_scope is not None:
        full["citation_scope"] = list(citation_scope)
    if turn.query_context is not None:
        full["query_context"] = _jsonable(turn.query_context)
    for path, record in ((official_path, official_record(turn, answer)), (audit_path, full)):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
