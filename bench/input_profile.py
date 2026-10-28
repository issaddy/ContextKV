"""Real-tokenizer P1 input budgets through the shared candidate assembly.

No inference, retrieval or answer labels determine eligibility. Export whole
eligible tasks and account for every excluded task; never truncate their input.
P2/P3 depend on actual generated history and are not predicted by this audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

from adapters.corpus import CorpusIndex
from adapters.mtrag import dataset_identity, load_rows, online_turn
from adapters.retrieval import to_evidence
from controller.candidates import build_candidates
from controller.config import create_model_template, load_config
from controller.schema import SessionSnapshot, TurnRecord
from controller.templates import HuggingFaceTemplate


def profile_input(row, *, corpus_id, snapshot_id, policy, template, gpu_capacity_tokens, corpus=None):
    turn = online_turn(row, "P1")
    history = tuple(TurnRecord(i//2+1, turn.history[i].text, turn.history[i+1].text)
                    for i in range(0, len(turn.history), 2))
    snapshot = SessionSnapshot(turn.conversation_id, raw_history=history,
                               corpus_snapshot_id=snapshot_id)
    priority = corpus.retrieved_keys(turn.contexts) if corpus is not None else None
    evidence = (corpus.resolve(priority) if corpus is not None else
                tuple(to_evidence(c, snapshot_id, corpus_id=corpus_id) for c in turn.contexts))
    candidate, = build_candidates(snapshot, turn.query, evidence, policy, template,
                                   retrieval_priority=priority)
    required = len(candidate.input_ids) + policy.max_new_tokens
    reason = candidate.reason if not candidate.feasible else (
        "input_plus_output_exceeds_gpu_pool" if required > gpu_capacity_tokens else None)
    return {"task_id": turn.task_id, "conversation_id": turn.conversation_id,
            "turn": turn.turn, "collection": turn.collection,
            "input_tokens": len(candidate.input_ids), "input_hash": candidate.input_hash,
            "input_plus_output_tokens": required, "given_sources": len(turn.contexts),
            "required_sources": len(candidate.required_blocks), "eligible": reason is None, "reason": reason}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-path", help="Local metadata copy; must match the configured model fingerprint")
    parser.add_argument("--evidence-corpus", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.model.scope is None:
        raise ValueError("input coverage requires an explicit real model scope, not byte-tokenizer simulation")
    template = (HuggingFaceTemplate(args.tokenizer_path, None, model_identity=config.model.scope.identity,
                                   local_files_only=True, enable_thinking=False) if args.tokenizer_path
                else create_model_template(config.model))
    config.model.scope.verify_fingerprint(asdict(template.fingerprint))
    rows = load_rows(args.input)
    if not rows:
        raise ValueError("input contains no tasks")
    identity = dataset_identity(rows)
    snapshot_id = identity["name"]+"-"+identity["revision"]
    corpus = (CorpusIndex.from_jsonl(args.evidence_corpus, corpus_id=identity["name"], snapshot_id=snapshot_id)
              if args.evidence_corpus else None)
    args.output.mkdir(parents=True, exist_ok=True)
    profiles, eligible = [], []
    with (args.output/"tasks.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            record = profile_input(row, corpus_id=identity["name"], snapshot_id=snapshot_id,
                                   policy=config.context, template=template,
                                   gpu_capacity_tokens=config.engine.gpu_capacity_tokens, corpus=corpus)
            profiles.append(record)
            if record["eligible"]:
                eligible.append(row["conversation_id"])
                output.write(json.dumps(row, ensure_ascii=False)+"\n")
    lengths = sorted(r["input_tokens"] for r in profiles)
    report = {"protocol": "independent-P1-input-budget-v1", "model_calls": 0,
              "source_sha256": sha256(args.input.read_bytes()).hexdigest(),
              "config_sha256": sha256(args.config.read_bytes()).hexdigest(),
              "corpus_sha256": sha256(args.evidence_corpus.read_bytes()).hexdigest() if args.evidence_corpus else None,
              "fingerprint": asdict(template.fingerprint), "policy": asdict(config.context),
              "gpu_capacity_tokens": config.engine.gpu_capacity_tokens,
              "requested_tasks": len(rows), "eligible_tasks": len(eligible),
              "excluded_tasks": len(rows)-len(eligible),
              "input_tokens": {"minimum": lengths[0], "median": median(lengths),
                               "mean": mean(lengths), "maximum": lengths[-1]},
              "input_at_least": {str(n): sum(v >= n for v in lengths) for n in (8192, 16384, 24576, 32768)},
              "exclusion_reasons": dict(Counter(r["reason"] for r in profiles if not r["eligible"])),
              "eligible_input_sha256": sha256((args.output/"tasks.jsonl").read_bytes()).hexdigest(),
              "rows": profiles,
              "scope": "Exact complete P1 inputs with configured output reservation and GPU token capacity; no generation, quality, concurrency or P2/P3 performance claim"}
    split = {"dev": [], "test": sorted(set(eligible)),
             "scope": "whole eligible P1 tasks in the paired tasks.jsonl; all excluded task IDs/reasons remain in input-profile.json"}
    for name, value in (("input-profile.json", report), ("split_manifest.json", split)):
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("requested_tasks", "eligible_tasks", "excluded_tasks",
                                           "input_tokens", "exclusion_reasons")}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
