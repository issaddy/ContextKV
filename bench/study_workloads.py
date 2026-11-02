"""Answer-driven development conversations and frozen request-arrival plans.

User-policy predicates inspect model answers; they do not score answer correctness.
References and goal conditions remain evaluation metadata, outside online questions.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re
import time

from controller.schema import TurnRecord


@dataclass(frozen=True)
class AnswerPredicate:
    all_terms: tuple[str, ...] = ()
    any_terms: tuple[str, ...] = ()
    excluded_terms: tuple[str, ...] = ()
    has_number: bool = False

    def matches(self, answer):
        text = answer.casefold()
        return (all(term.casefold() in text for term in self.all_terms)
                and (not self.any_terms or any(term.casefold() in text for term in self.any_terms))
                and not any(term.casefold() in text for term in self.excluded_terms)
                and (not self.has_number or re.search(r"\d", text) is not None))


@dataclass(frozen=True)
class GoalEdge:
    branch: str
    target: str
    predicate: AnswerPredicate = field(default_factory=AnswerPredicate)


@dataclass(frozen=True)
class GoalNode:
    goal_id: str
    question: str
    edges: tuple[GoalEdge, ...] = ()
    conditions: tuple[tuple[str, str], ...] = ()
    offline_reference: dict | None = None


@dataclass(frozen=True)
class GoalGraph:
    graph_id: str
    initial_goal: str
    nodes: tuple[GoalNode, ...]
    turns: int
    source_document_ids: tuple[str, ...]
    origin: str = "constructed_dev_from_native_training_sources"


@dataclass(frozen=True)
class QuestionDecision:
    question: str
    goal_id: str
    branch: str
    metadata: dict


class GoalGraphPolicy:
    """Replay the fixed graph over committed model QA, including on resume."""
    def __init__(self, graphs):
        self.graphs = {graph.graph_id: graph for graph in graphs}

    def write(self, path):
        Path(path).write_text(json.dumps({"schema_version": 1, "evaluation_only": True,
            "graphs": [asdict(graph) for graph in self.graphs.values()]}, ensure_ascii=False, indent=2)+"\n")

    def question_fn(self, row, actual_history: tuple[TurnRecord, ...]):
        graph = self.graphs[row["goal_graph_id"]]
        nodes = {node.goal_id: node for node in graph.nodes}
        current, branch = graph.initial_goal, "initial"
        path = []
        for turn in actual_history:
            edge = next(edge for edge in nodes[current].edges if edge.predicate.matches(turn.assistant_text))
            path.append({"goal_id": current, "branch": edge.branch, "next_goal_id": edge.target,
                         "answer_turn_id": turn.turn_id})
            current, branch = edge.target, edge.branch
        node = nodes[current]
        return QuestionDecision(node.question, current, branch, {
            "dataset": "dynamic_goal_dev", "protocol": "P3_answer_driven_goal_graph",
            "graph_id": graph.graph_id, "goal_id": current, "branch": branch,
            "goal_conditions": dict(node.conditions), "goal_path": path,
            "offline_reference": node.offline_reference,
            "source_document_ids": graph.source_document_ids,
            "policy_origin": graph.origin,
            "actual_history": [{"turn_id": turn.turn_id, "user_text": turn.user_text,
                                "assistant_text": turn.assistant_text} for turn in actual_history]})


def load_goal_policy(path, *, expected_sha256=None):
    payload = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("goal policy changed after workload freeze")
    graphs = []
    for graph in json.loads(payload)["graphs"]:
        nodes = []
        for node in graph["nodes"]:
            edges = tuple(GoalEdge(edge["branch"], edge["target"], AnswerPredicate(**{
                name: tuple(value) if isinstance(value, list) else value
                for name, value in edge["predicate"].items()})) for edge in node["edges"])
            nodes.append(GoalNode(**{**node, "edges": edges, "conditions": tuple(map(tuple, node["conditions"]))}))
        graphs.append(GoalGraph(**{**graph, "nodes": tuple(nodes), "source_document_ids": tuple(graph["source_document_ids"])}))
    return GoalGraphPolicy(graphs)


def build_goal_graph_dataset(version_rows, *, max_conversations=None, topic_request="summary"):
    """Use public constructed-dev questions/metadata; never read their targets."""
    if topic_request not in {"summary", "allowance_lookup"}:
        raise ValueError("topic_request must be summary or allowance_lookup")
    first = {row["conversation_id"]: row for row in version_rows if row["turn"] == 1}
    all_conversations = sorted(first)
    conversations = all_conversations
    if max_conversations is not None:
        conversations = conversations[:max_conversations]
    graphs, rows = [], []
    for index, conversation in enumerate(conversations):
        original = first[conversation]
        own = original["contexts"]
        versions = sorted({(doc["valid_from"], doc["version"]) for doc in own})
        old, current = versions[0][1], versions[-1][1]
        other = first[all_conversations[(index+1) % len(all_conversations)]]
        topic = (max(other["contexts"], key=lambda doc: (
            doc["valid_from"], doc["version"], doc["applicability"]["PLATFORM"]))
            if topic_request == "allowance_lookup" else other["contexts"][0])
        topic_question = (f"Switch to the supplied source titled {topic['title']!r}. "
            + (f"Under policy version {topic['version']}, what filing allowance does it specify "
               f"for {topic['applicability']['PLATFORM']}?" if topic_request == "allowance_lookup" else
               "Describe the subject of that source in one paragraph."))
        topic_conditions = (("TOPIC", topic["document_family"]),)
        if topic_request == "allowance_lookup":
            topic_conditions += (("PLATFORM", topic["applicability"]["PLATFORM"]),
                                 ("VERSION", topic["version"]), ("REQUEST_KIND", topic_request))
        graph_id = f"dynamic-goal:{conversation}"
        conditions = (("PLATFORM", "Linux"),)
        nodes = (
            GoalNode("current_windows", original["input"][-1]["text"], (
                GoalEdge("windows_acknowledged", "correct_linux", AnswerPredicate(all_terms=("Windows",), has_number=True)),
                GoalEdge("platform_needs_clarification", "clarify_linux")), (("PLATFORM", "Windows"), ("VERSION", "current"))),
            GoalNode("correct_linux", "Change the platform to Linux. What is its current allowance?", (
                GoalEdge("linux_acknowledged", "previous_policy", AnswerPredicate(all_terms=("Linux",), has_number=True)),
                GoalEdge("linux_still_missing", "clarify_previous")), conditions),
            GoalNode("clarify_linux", "Use Linux for the same program, replacing Windows. What is its current allowance?", (
                GoalEdge("linux_acknowledged", "previous_policy", AnswerPredicate(all_terms=("Linux",), has_number=True)),
                GoalEdge("linux_still_missing", "clarify_previous")), conditions),
            GoalNode("previous_policy", "What was its Linux allowance immediately before that current policy?", (
                GoalEdge("previous_version_named", "compare", AnswerPredicate(all_terms=(old,))),
                GoalEdge("previous_version_not_named", "clarify_old")), (*conditions, ("VERSION_RELATION", "before"))),
            GoalNode("clarify_previous", "For Linux in the original program, what allowance applied immediately before the current policy?", (
                GoalEdge("previous_version_named", "compare", AnswerPredicate(all_terms=(old,))),
                GoalEdge("previous_version_not_named", "clarify_old")), (*conditions, ("VERSION_RELATION", "before"))),
            GoalNode("compare", f"Compare the Linux allowances in policy versions {old} and {current}.", (
                GoalEdge("comparison_received", "remove_platform"),), (*conditions, ("VERSIONS", f"{old},{current}"))),
            GoalNode("clarify_old", f"Use policy version {old} explicitly, then compare its Linux allowance with version {current}.", (
                GoalEdge("explicit_version_requested", "remove_platform"),), (*conditions, ("VERSIONS", f"{old},{current}"))),
            GoalNode("remove_platform", "Remove the platform restriction. Under the current policy, give the allowance for every supplied platform.", (
                GoalEdge("platform_restriction_removed", "topic_switch"),), (("PLATFORM", "removed"), ("VERSION", "current"))),
            GoalNode("topic_switch", topic_question, (GoalEdge("topic_answer_received", "source_revisit"),),
                topic_conditions),
            GoalNode("source_revisit", "Return to the original program. Restore Linux and the current policy: what is its allowance?", (),
                (*conditions, ("VERSION", "current"))),
        )
        documents = {doc["document_id"]: doc for doc in (*own, *other["contexts"])}
        graph = GoalGraph(graph_id, "current_windows", nodes, 7, tuple(documents))
        graphs.append(graph)
        for turn in range(1, graph.turns+1):
            rows.append({"dataset": {"name": "dynamic_goal_dev", "revision":
                "goal-graph-v2-allowance-topic" if topic_request == "allowance_lookup" else "goal-graph-v1"},
                "conversation_id": graph_id, "task_id": f"{graph_id}:{turn}", "turn": turn,
                "Collection": "dynamic_goal_dev/constructed_dev", "split": "dev", "goal_graph_id": graph_id,
                "input": [{"speaker": "user", "text": nodes[0].question if turn == 1 else ""}],
                "contexts": list(documents.values()), "coverage_mode": "selected_sources",
                "workload_topic": topic["document_family"] if turn == 6 else own[0]["document_family"],
                "workload_source_document_ids": [doc["document_id"] for doc in (other["contexts"] if turn == 6 else own)],
                "label_origin": graph.origin})
    return GoalGraphPolicy(graphs), rows


@dataclass(frozen=True)
class WorkloadTurn:
    task_id: str
    conversation_id: str
    turn: int
    dispatch_index: int
    release_after_task_id: str | None
    think_time_seconds: float
    start_delay_seconds: float
    topic_id: str
    source_document_ids: tuple[str, ...]
    topic_switch: bool
    source_revisit: bool


@dataclass(frozen=True)
class WorkloadManifest:
    name: str
    concurrency: int
    turns: tuple[WorkloadTurn, ...]
    source_rows_sha256: str
    goal_policy_sha256: str | None = None
    schema_version: int = 1
    concurrency_scope: str = "request"
    time_origin: str = "dispatcher construction: monotonic seconds"
    dispatch_semantics: str = "explicit ordinal after own prior commit plus fixed think-time"

    @property
    def dispatch_order(self):
        return tuple(turn.task_id for turn in self.turns)

    def think_time_fn(self, row, actual_history):
        return next(turn.think_time_seconds for turn in self.turns if turn.task_id == row["task_id"])

    def write(self, path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2)+"\n")


def read_workload_manifest(path):
    value = json.loads(Path(path).read_text())
    value["turns"] = tuple(WorkloadTurn(**{**turn, "source_document_ids": tuple(turn["source_document_ids"])})
                            for turn in value["turns"])
    return WorkloadManifest(**value)


def freeze_workload(rows, *, name, concurrency, think_time_seconds=0.0, pause_after=None,
                    session_start_delays=None, interleaving=None, goal_policy_sha256=None):
    """Keep questions/references fixed; freeze arrivals and annotate source reuse."""
    pause_after, session_start_delays = pause_after or {}, session_start_delays or {}
    grouped = {}
    for row in rows:
        grouped.setdefault(row["conversation_id"], []).append(row)
    ordered = {sid: sorted(turns, key=lambda row: row["turn"]) for sid, turns in sorted(grouped.items())}
    by_task = {row["task_id"]: row for row in rows}
    order = (tuple(interleaving) if interleaving is not None else tuple(
        turns[index]["task_id"] for index in range(max(map(len, ordered.values())))
        for turns in ordered.values() if index < len(turns)))
    if len(order) != len(by_task) or set(order) != set(by_task):
        raise ValueError("interleaving must contain every fixed task exactly once")
    index = {task_id: position for position, task_id in enumerate(order)}
    entries = {}
    for sid, turns in ordered.items():
        seen, previous, previous_topic, previous_sources = set(), None, None, set()
        for row in turns:
            sources = tuple(row["workload_source_document_ids"] if "workload_source_document_ids" in row
                            else (doc["document_id"] for doc in row.get("contexts", ())))
            topic = row.get("workload_topic", row.get("goal_graph_id", row["Collection"]))
            if previous is not None and index[previous] >= index[row["task_id"]]:
                raise ValueError("interleaving must preserve committed order within each conversation")
            entries[row["task_id"]] = WorkloadTurn(row["task_id"], sid, row["turn"], index[row["task_id"]],
                previous, 0.0 if previous is None else pause_after.get(previous, think_time_seconds),
                session_start_delays.get(sid, 0.0) if previous is None else 0.0, topic, sources,
                previous_topic is not None and previous_topic != topic,
                bool((seen-previous_sources).intersection(sources)))
            seen.update(sources)
            previous, previous_topic = row["task_id"], topic
            previous_sources = set(sources)
    frozen = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return WorkloadManifest(name, concurrency, tuple(entries[task_id] for task_id in order),
                            hashlib.sha256(frozen).hexdigest(), goal_policy_sha256)


def prepare_workloads(config, matrix):
    """Freeze one constructed-dev goal cohort and six arrival schedules locally."""
    import yaml
    output = Path(config["study"]["output_dir"])
    source = Path(matrix["datasets"]["versionqa"]["inputs"]["dev"])
    original = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    policy, rows = build_goal_graph_dataset(original,
        max_conversations=config.get("workloads", {}).get("conversations"),
        topic_request=config.get("workloads", {}).get("topic_request", "summary"))
    by_conversation = {}
    for row in original:
        by_conversation.setdefault(row["conversation_id"], {})[row["turn"]] = row
    referenced = []
    for graph in policy.graphs.values():
        original_id = graph.graph_id.removeprefix("dynamic-goal:")
        references = by_conversation[original_id]
        turns_by_goal = {"current_windows": (1,), "correct_linux": (2,), "clarify_linux": (2,),
            "previous_policy": (3,), "clarify_previous": (3,), "compare": (4,), "clarify_old": (4,),
            "remove_platform": (1, 2), "source_revisit": (2,)}
        nodes = []
        for node in graph.nodes:
            if node.goal_id == "topic_switch":
                family = dict(node.conditions)["TOPIC"]
                conditions = dict(node.conditions)
                source_doc = next(doc for row in original for doc in row["contexts"]
                    if doc["document_family"] == family and (conditions.get("REQUEST_KIND") != "allowance_lookup" or
                    (doc["version"] == conditions["VERSION"] and
                     doc["applicability"]["PLATFORM"] == conditions["PLATFORM"])))
                reference = {"source_document_ids": [source_doc["document_id"]], "source_text": source_doc["text"],
                             "origin": "public constructed-dev source text"}
            else:
                reference = {"answers": [target["text"] for turn in turns_by_goal[node.goal_id]
                    for target in references[turn]["targets"]],
                    "origin": "constructed-dev references; evaluation only"}
            nodes.append(replace(node, offline_reference=reference))
        referenced.append(replace(graph, nodes=tuple(nodes)))
    policy = GoalGraphPolicy(referenced)
    destination = output/"evaluation"/"dynamic_goal_dev"
    destination.mkdir(parents=True, exist_ok=True)
    tasks_path, graphs_path = destination/"tasks.jsonl", destination/"goal_graphs.json"
    tasks_path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows))
    policy.write(graphs_path)
    goal_policy_sha256 = hashlib.sha256(graphs_path.read_bytes()).hexdigest()
    arrivals = output/"workloads"
    arrivals.mkdir(parents=True, exist_ok=True)
    schedules = []
    for concurrency in (1, 4, 8):
        for pause in (0, 30):
            name = f"dynamic-goal-c{concurrency}-pause{pause}"
            path = arrivals/f"{name}.json"
            freeze_workload(rows, name=name, concurrency=concurrency,
                pause_after={row["task_id"]: float(pause) for row in rows if row["turn"] == 3},
                goal_policy_sha256=goal_policy_sha256).write(path)
            schedules.append({"name": name, "concurrency": concurrency, "pause_after_turn": 3,
                              "pause_seconds": pause, "path": str(path.resolve())})
    diagnostic = {**matrix, "protocol": "P3_answer_driven_goal_graph",
        "selection": {**matrix.get("selection", {}), "development_weights": {"dynamic_goal_dev": 1.0}},
        "datasets": {"dynamic_goal_dev": {"inputs": {"dev": str(tasks_path.resolve())},
            "split_kind": "constructed_native_training_source_dev_goal_graph", "protocol": "P3",
            "supplied_documents": True, "goal_policy": str(graphs_path.resolve()),
            "reference_mode": "per_record_question_decision_offline_reference",
            "primary_metric": "project_condition_version_score", "normalization": 1.0, "weight": 1.0,
            "diagnostic_only": True, "include_in_public_quality_aggregate": False}},
        "cohorts": {"dev": {"dynamic_goal_dev": [row["task_id"] for row in rows]}},
        "workload_schedules": schedules, "public_quality_aggregate": "excluded"}
    matrix_path = arrivals/"diagnostic_matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(diagnostic, allow_unicode=True, sort_keys=False))
    manifest = {"schema_version": 1, "dataset": "dynamic_goal_dev", "split": "dev",
        "topic_request": config.get("workloads", {}).get("topic_request", "summary"),
        "conversations": len(policy.graphs), "turns": len(rows), "turns_per_conversation": 7,
        "tasks_path": str(tasks_path.resolve()), "goal_policy_path": str(graphs_path.resolve()),
        "goal_policy_sha256": goal_policy_sha256,
        "diagnostic_matrix_path": str(matrix_path.resolve()), "schedules": schedules,
        "source_tasks": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "phase_analysis": ["condition_correction", "version_before", "version_compare", "condition_remove",
                           "topic_switch", "source_revisit"],
        "online_policy_input": "actual committed model QA and public source metadata",
        "offline_references": "evaluation metadata only; not passed to retrieval or task interpretation",
        "public_test_assets_read": [], "public_quality_aggregate": "excluded"}
    (arrivals/"prepare_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest


class WorkloadDispatcher:
    """Arrival gate independent of the runner's in-flight request semaphore."""
    def __init__(self, manifest):
        self.manifest = manifest
        self.origin = time.monotonic()
        self._turns = {turn.task_id: turn for turn in manifest.turns}
        self._completed = {task_id: asyncio.Event() for task_id in self._turns}
        self._completion_times = {}
        self._condition = asyncio.Condition()
        self._next = 0
        self.arrivals = {}
        self.cancelled = {}
        self.restored_task_ids = ()

    def restore_completed(self, task_ids):
        """Start a new measured segment; reapply pauses from its time origin."""
        self.restored_task_ids = tuple(task_ids)
        for task_id in self.restored_task_ids:
            self._completion_times[task_id] = 0.0
            self._completed[task_id].set()
        self._skip_completed_ordinals()

    def _skip_completed_ordinals(self):
        order = self.manifest.dispatch_order
        while self._next < len(order) and (self._completed[order[self._next]].is_set()
                                          or order[self._next] in self.cancelled):
            self._next += 1

    async def wait_turn(self, row, actual_history):
        turn = self._turns[row["task_id"]]
        if turn.release_after_task_id is None:
            planned = turn.start_delay_seconds
        else:
            await self._completed[turn.release_after_task_id].wait()
            planned = self._completion_times[turn.release_after_task_id]+turn.think_time_seconds
        await asyncio.sleep(max(0.0, self.origin+planned-time.monotonic()))
        async with self._condition:
            independent = self.manifest.dispatch_semantics == "independent_clients_after_own_prior_commit"
            if not independent:
                await self._condition.wait_for(lambda: turn.dispatch_index == self._next)
            arrival = time.monotonic()-self.origin
            value = {"task_id": turn.task_id, "dispatch_index": turn.dispatch_index,
                "release_after_task_id": turn.release_after_task_id,
                "fixed_think_time_seconds": turn.think_time_seconds,
                "planned_arrival_seconds": planned, "actual_arrival_seconds": arrival,
                "dispatch_delay_seconds": arrival-planned,
                "measurement_segment_restored_tasks": self.restored_task_ids,
                "resume_pause_basis": "configured pause reapplied from segment origin" if self.restored_task_ids else None,
                "committed_history_turns": len(actual_history)}
            self.arrivals[turn.task_id] = value
            if not independent:
                self._next += 1
            self._skip_completed_ordinals()
            self._condition.notify_all()
            return value

    async def cancel_conversation(self, conversation_id, reason):
        """Remove unsubmitted ordinals after a failed client, without commits."""
        async with self._condition:
            for turn in self.manifest.turns:
                if (turn.conversation_id == conversation_id and turn.task_id not in self.arrivals
                        and not self._completed[turn.task_id].is_set()):
                    self.cancelled[turn.task_id] = {"reason": reason, "completed": False}
            self._skip_completed_ordinals()
            self._condition.notify_all()

    def complete_turn(self, task_id):
        self._completion_times[task_id] = time.monotonic()-self.origin
        self._completed[task_id].set()
