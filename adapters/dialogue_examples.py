"""Offline causal next-agent examples from the Doc2Dial supervision export.

Inputs contain only original turns strictly before the target. Given-document
identity is an explicit offline task scope, not a retrieval result. Original
author targets and labels are supervision, not verified quality/action labels.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path


def project_conversation(conversation):
    """Yield one independent example per original agent turn, without alternation.

    The exported source already validates reference coordinates. This projection
    does not reinterpret them or infer semantic labels from dialogue acts.
    """
    source_format = conversation["source_format"]
    if source_format not in {"v1", "v0.9"}:
        raise ValueError("source_format must be v1 or v0.9 from the Doc2Dial export")
    reference_field = "references" if source_format == "v1" else "reference"
    turns = conversation["turns"]
    for turn in turns:
        if turn["role"] not in {"user", "agent"}:
            raise ValueError(f"unsupported original role: {turn['role']!r}")

    prefix = []
    for turn in turns:
        if turn["role"] == "agent":
            yield {
                **{key: conversation[key] for key in
                   ("dataset", "dataset_version", "source_format", "split")},
                "address": {
                    "conversation_id": conversation["conversation_id"],
                    "source_dialogue_id": conversation["source_dialogue_id"],
                    "target_turn_id": turn["turn_id"],
                },
                "task_scope": {
                    "kind": "offline_given_document",
                    **{key: conversation[key] for key in
                       ("passage_id", "domain", "source_document_id")},
                },
                "input": {"turns": [item.copy() for item in prefix]},
                "supervision": deepcopy({key: turn[key] for key in
                    ("turn_id", "role", "utterance", "da", reference_field, "grounding")}),
            }
        prefix.append({key: turn[key] for key in ("turn_id", "role", "utterance")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervision", type=Path, required=True,
                        help="supervision.jsonl produced by adapters.doc2dial")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; preserve earlier exports")

    digest = sha256()
    counts = {"conversations": 0, "turns": 0, "examples": 0,
              "empty_prefix_examples": 0, "consecutive_agent_targets": 0,
              "terminal_user_conversations": 0}
    partitions = {}
    args.output.mkdir(parents=True, exist_ok=False)
    with args.supervision.open("rb") as source, (args.output / "examples.jsonl").open("x", encoding="utf-8") as output:
        for line in source:
            digest.update(line)
            conversation = json.loads(line)
            partition = tuple(conversation[key] for key in
                              ("dataset", "dataset_version", "source_format", "split"))
            partitions.setdefault(partition, {"conversations": 0, "examples": 0})
            turns = conversation["turns"]
            counts["conversations"] += 1
            counts["turns"] += len(turns)
            counts["terminal_user_conversations"] += bool(turns and turns[-1]["role"] == "user")
            partitions[partition]["conversations"] += 1
            for example in project_conversation(conversation):
                output.write(json.dumps(example, ensure_ascii=False) + "\n")
                prefix = example["input"]["turns"]
                counts["examples"] += 1
                counts["empty_prefix_examples"] += not prefix
                counts["consecutive_agent_targets"] += bool(prefix and prefix[-1]["role"] == "agent")
                partitions[partition]["examples"] += 1

    manifest = {
        "status": "completed", "projection": "causal_next_agent", **counts,
        "source": {"path": str(args.supervision.resolve()), "sha256": digest.hexdigest()},
        "partitions": [dict(zip(("dataset", "dataset_version", "source_format", "split"), key), **value)
                       for key, value in partitions.items()],
        "input_scope": "strict_preceding_original_turns_role_utterance_id",
        "task_scope": "offline_given_document_not_live_retrieval",
        "supervision_scope": "original_author_target_not_verified_quality_or_action_label",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "conversations", "turns", "examples")}))
    return manifest


if __name__ == "__main__":
    main()
