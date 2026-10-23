"""Replay an optional correction-aware query selector on frozen dev retrieval.

Routing reads only user questions. Qrels enter only after every row is selected;
saved retrieval times are not new end-to-end latency measurements.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from adapters.hybrid_retrieval import query_text
from adapters.query_views import correction_cue, correction_query_mode
from bench.quality import load_qrels
from bench.retrieval_eval import summarize


def select_rows(rows, tasks):
    lookup = {(r["task_id"], r["stage"], r["query_mode"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate frozen task/stage/query-mode row")
    stages = sorted({r["stage"] for r in rows})
    selected, routes = [], []
    for task in tasks:
        questions = task["user_questions"]
        if not questions:
            raise ValueError("each frozen task needs its current user question")
        # Route is fixed before selecting saved result IDs or reading labels.
        mode = correction_query_mode(questions[-1])
        cue = correction_cue(questions[-1])
        routes.append({"task_id": task["task_id"], "conversation_id": task["conversation_id"],
                       "turn": task["turn"], "current_question": questions[-1],
                       "selected_query_mode": mode, "cue": cue,
                       "visible_user_questions": questions})
        for stage in stages:
            for candidate_mode in ("current", "last3"):
                source = lookup.get((task["task_id"], stage, candidate_mode))
                if source is None or source["status"] != "complete":
                    raise ValueError("every task/stage needs complete current and last3 frozen results")
                if source["query_text"] != query_text(questions, candidate_mode):
                    raise ValueError("frozen query text differs from its user-only view")
            source = lookup[task["task_id"], stage, mode]
            selected.append({**source, "query_mode": "correction_aware", "method": f"correction_aware/{stage}",
                             "selected_query_mode": mode, "routing_cue": cue,
                             "source_method": source["method"], "operation": "offline_frozen_result_selection"})
    return selected, routes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--output", required=True, help="new development exploration directory")
    args = parser.parse_args(argv)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("choose a new selector exploration directory")
    source = Path(args.retrieval)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest["config"]["split"] != "dev" or manifest["status"] != "complete":
        raise ValueError("selector exploration requires a complete frozen dev retrieval run")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    selected, routes = select_rows(rows, manifest["tasks"])
    # This boundary occurs after routes are immutable. No reference answers are loaded.
    qrels_path = Path(args.qrels)
    qrels_digest = sha256(qrels_path.read_bytes()).hexdigest()
    if qrels_digest != manifest["sources"]["qrels"]["sha256"]:
        raise ValueError("use the same dev qrels as the frozen retrieval run")
    summary = summarize(rows + selected, load_qrels(qrels_path), candidate_k=manifest["config"]["candidate_k"])
    summary.update(scope="case-inspired dev exploration; not independent heldout validation",
        selector="current-user explicit correction cue -> last3, otherwise current",
        routing_uses_task_ids_or_labels=False, routing_uses_assistant_answers=False,
        triggered_queries=[r for r in routes if r["cue"] is not None], routing=routes,
        new_retrieval_calls=0, new_generation_calls=0,
        timing_scope="selected saved stage elapsed times only; no new pipeline latency measurement",
        provenance={"retrieval": {"path": str(source.resolve()), "sha256": sha256(source.read_bytes()).hexdigest()},
                    "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path.read_bytes()).hexdigest()},
                    "qrels": {"path": str(qrels_path.resolve()), "sha256": qrels_digest}})
    target.mkdir(parents=True)
    (target / "selected-retrieval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(target), "triggered": len(summary["triggered_queries"]),
        "methods": {name: {k: value["top5"][k] for k in ("mean_recall", "mean_ndcg", "evaluated_tasks")}
                    for name, value in summary["methods"].items()}}, indent=2))


if __name__ == "__main__":
    main()
