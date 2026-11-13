"""MT-RAG source loading and offline dataset preparation."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

DATASET_COMMIT = "53fef6f5917c7e10c436f697e69a955dbc795d81"
DATASET_SHA256 = "5d5201da9fabd072fd8f6b8d051bfaafa7ef031e76722a4920c66e94cede1873"
DATASET_URL = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/{DATASET_COMMIT}/mtrag-human/generation_tasks/RAG.jsonl"
MTRAG_UN_URL = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/{DATASET_COMMIT}/mtragun-human/generation_tasks/reference.jsonl"
MTRAG_UN_SHA256 = "68053f709c928570718fd2c88455eccf974746f5bfa2206d7fd39b969acb21d4"
CLOUD_COLLECTION = "mt-rag-ibmcloud-elser-512-100-20240502"


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Offline-only loader; do not hand these dictionaries to a planner."""
    data = Path(path).read_bytes()
    # Official files use descriptive string labels in `dataset`, not the
    # canonical adapter identity mapping. Bind only the pinned file bytes;
    # a copied label in an edited/custom file cannot assert this revision.
    name = {DATASET_SHA256: "mtrag", MTRAG_UN_SHA256: "mtrag-un"}.get(sha256(data).hexdigest())
    rows = []
    seen = set()
    for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("conversation_id", "task_id", "turn", "Collection", "input", "contexts"):
            if key not in row:
                raise ValueError(f"line {line_no}: missing {key}")
        identity = (str(row["conversation_id"]), int(row["turn"]))
        if identity in seen:
            raise ValueError(f"duplicate conversation/turn: {identity}")
        seen.add(identity)
        if name is not None:
            row["dataset"] = {"name": name, "revision": DATASET_COMMIT, "commit": DATASET_COMMIT}
        rows.append(row)
    return rows


def dataset_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Read an adapter's declared identity; legacy MT-RAG files keep theirs."""
    legacy = {"name": "mtrag", "revision": DATASET_COMMIT, "commit": DATASET_COMMIT}
    value = rows[0].get("dataset", legacy) if rows else legacy
    if (not isinstance(value, Mapping)
            or any(not isinstance(value.get(k), str) or not value[k].strip() for k in ("name", "revision"))):
        raise ValueError("dataset identity requires name/revision metadata; use load_rows for pinned official files")
    identity = dict(value)
    if any(row.get("dataset", legacy) != identity for row in rows):
        raise ValueError("one benchmark job must use one dataset identity")
    return identity


def group_sessions(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    sessions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        sessions[str(row["conversation_id"])].append(row)
    return {sid: sorted(turns, key=lambda r: int(r["turn"])) for sid, turns in sorted(sessions.items())}


def _evidence_key(context: Mapping[str, Any]) -> tuple[str, str]:
    return str(context["document_id"]), sha256(str(context["text"]).encode()).hexdigest()


def profile_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sessions = group_sessions(rows)
    total = adjacent = historical = revisit = maximum = 0
    for turns in sessions.values():
        maximum = max(maximum, len(turns))
        previous: set[tuple[str, str]] = set()
        seen: set[tuple[str, str]] = set()
        for index, row in enumerate(turns):
            current = {_evidence_key(c) for c in row["contexts"]}
            if index:
                total += len(current)
                adjacent += len(current & previous)
                historical += len(current & seen)
                revisit += len((current & seen) - previous)
            previous = current
            seen |= current
    return {"sessions": len(sessions), "tasks": len(rows), "max_turns": maximum,
            "max_input_turn": max((int(row["turn"]) for row in rows), default=0),
            "current_blocks_after_first": total, "adjacent_repeat": adjacent,
            "ever_repeat": historical, "nonadjacent_revisit": revisit,
            "adjacent_ratio": adjacent / total if total else 0.0,
            "ever_ratio": historical / total if total else 0.0}


def dataset_profile(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identity = ({"dataset": dataset_identity(rows)} if rows and "dataset" in rows[0]
                else {"dataset_commit": DATASET_COMMIT})
    return {**identity, "all": profile_rows(rows),
            "cloud": profile_rows([r for r in rows if r["Collection"] in {CLOUD_COLLECTION, "ibmcloud"}]),
            "reference_markers": sum("reference" in c for r in rows for c in r["contexts"])}


def json_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def stratified_split(rows: Sequence[Mapping[str, Any]], seed: int = 42,
                     dev_fraction: float = .2, source_sha256: str | None = None) -> dict[str, Any]:
    """Freeze explicit session IDs; stratify by domain without splitting conversations."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    groups: dict[str, list[str]] = defaultdict(list)
    for session_id, turns in group_sessions(rows).items():
        domains = {str(r["Collection"]) for r in turns}
        if len(domains) != 1:
            raise ValueError("a conversation cannot span collections")
        groups[domains.pop()].append(session_id)
    rng = random.Random(seed)
    dev, test, strata = [], [], {}
    for domain, ids in sorted(groups.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        count = min(len(ids) - 1, max(1, round(len(ids) * dev_fraction))) if len(ids) > 1 else 0
        dev += ids[:count]
        test += ids[count:]
        strata[domain] = {"dev": sorted(ids[:count]), "test": sorted(ids[count:])}
    identity = ({"dataset": dataset_identity(rows)} if rows and "dataset" in rows[0]
                else {"dataset_commit": DATASET_COMMIT})
    value = {**identity, "source_sha256": source_sha256, "seed": seed,
             "dev_fraction": dev_fraction, "unit": "conversation", "dev": sorted(dev),
             "test": sorted(test), "strata": strata}
    value["manifest_sha256"] = json_hash(value)
    return value


def split_overlap(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Document overlap is descriptive; topic labels remain offline, never online."""
    dev_ids, test_ids = set(manifest["dev"]), set(manifest["test"])
    if dev_ids & test_ids:
        raise ValueError("split leaks conversation IDs")
    dev_docs = {str(c["document_id"]) for r in rows if r["conversation_id"] in dev_ids for c in r["contexts"]}
    test_docs = {str(c["document_id"]) for r in rows if r["conversation_id"] in test_ids for c in r["contexts"]}
    return {"dev_documents": len(dev_docs), "test_documents": len(test_docs),
            "shared_documents": len(dev_docs & test_docs), "topic_overlap": None,
            "topic_overlap_note": "No frozen topic clustering is configured."}
