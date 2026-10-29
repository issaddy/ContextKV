"""Pinned MTRAG reader. Only newly constructed whitelist objects reach serving code.

Raw rows and split/profile utilities are offline infrastructure. OnlineSession exposes
one current turn at a time and requires a completed answer before advancing; P2/P3
never copy official assistant messages or annotations into online objects.
"""
from __future__ import annotations

from controller.schema import OnlineMessage, OnlineTurn
from adapters.retrieval import clean_context

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Iterable, Literal, Mapping, Sequence

DATASET_COMMIT = "53fef6f5917c7e10c436f697e69a955dbc795d81"
DATASET_SHA256 = "5d5201da9fabd072fd8f6b8d051bfaafa7ef031e76722a4920c66e94cede1873"
DATASET_URL = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/{DATASET_COMMIT}/mtrag-human/generation_tasks/RAG.jsonl"
MTRAG_UN_URL = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/{DATASET_COMMIT}/mtragun-human/generation_tasks/reference.jsonl"
MTRAG_UN_SHA256 = "68053f709c928570718fd2c88455eccf974746f5bfa2206d7fd39b969acb21d4"
CLOUD_COLLECTION = "mt-rag-ibmcloud-elser-512-100-20240502"
Protocol = Literal["P1", "P2", "P3"]


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


def _message(message: Mapping[str, Any] | OnlineMessage) -> OnlineMessage:
    if isinstance(message, OnlineMessage):
        return message
    speaker = str(message.get("speaker", message.get("role", ""))).lower()
    speaker = {"human": "user", "agent": "assistant"}.get(speaker, speaker)
    if speaker not in {"user", "assistant"}:
        raise ValueError(f"unsupported MTRAG speaker {speaker!r}")
    return OnlineMessage(speaker, str(message.get("text", message.get("content", ""))))


def online_turn(row: Mapping[str, Any], protocol: Protocol = "P2",
                generated_history: Sequence[OnlineMessage | Mapping[str, Any]] = ()) -> OnlineTurn:
    if protocol not in {"P1", "P2", "P3"}:
        raise ValueError("protocol must be P1, P2 or P3")
    turn_number = int(row["turn"])
    inputs = row["input"]
    if turn_number < 1 or len(inputs) != 2 * turn_number - 1:
        raise ValueError("task input must contain exactly the history up to its numbered turn")
    question = _message(inputs[-1])
    if question.speaker != "user":
        raise ValueError("task input must end at the current user question")
    history = (tuple(_message(m) for m in inputs[:-1]) if protocol == "P1"
               else tuple(_message(m) for m in generated_history))
    if len(history) != 2 * (turn_number - 1):
        raise ValueError("generated history must include every prior successfully committed turn")
    if len(history) % 2 or any(m.speaker != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(history)):
        raise ValueError("allowed history must contain complete alternating user/assistant pairs")
    # P3 has no access to even the official current evidence: its retriever fills this separately.
    contexts = () if protocol == "P3" else tuple(clean_context(c) for c in row["contexts"])
    return OnlineTurn(str(row["conversation_id"]), str(row["task_id"]), int(row["turn"]),
                      str(row["Collection"]), question.text, history, contexts, protocol)


class OnlineSession:
    """Infrastructure-owned sequential gate; consumers only receive current().

    A handler is passed the resulting OnlineTurn, never this gate or its raw rows.
    commit() is called only for a successful, persisted generation. A failed attempt
    can retry current(); it cannot reveal a subsequent question or evidence block.
    """
    def __init__(self, rows: Sequence[Mapping[str, Any]], protocol: Protocol = "P2"):
        self.__rows = tuple(sorted(rows, key=lambda r: int(r["turn"])))
        if not self.__rows:
            raise ValueError("session is empty")
        if len({str(r["conversation_id"]) for r in self.__rows}) != 1:
            raise ValueError("rows must belong to one conversation")
        turns = [int(r["turn"]) for r in self.__rows]
        if len(set(turns)) != len(turns):
            raise ValueError("duplicate turn")
        if protocol != "P1" and turns != list(range(1, len(turns) + 1)):
            raise ValueError("generated-history protocols need every turn from 1; selected-turn benchmarks use P1")
        self.__position = 0
        self.__history: tuple[OnlineMessage, ...] = ()
        self.protocol = protocol
        self.conversation_id = str(self.__rows[0]["conversation_id"])

    @property
    def done(self) -> bool:
        return self.__position == len(self.__rows)

    @property
    def remaining(self) -> int:
        return len(self.__rows) - self.__position

    def current(self) -> OnlineTurn:
        if self.done:
            raise StopIteration("session completed")
        return online_turn(self.__rows[self.__position], self.protocol, self.__history)

    def commit(self, task_id: str, assistant_text: str) -> None:
        current = self.current()
        if task_id != current.task_id:
            raise ValueError("only the current task may be committed")
        self.__history += (OnlineMessage("user", current.query), OnlineMessage("assistant", assistant_text))
        self.__position += 1


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
