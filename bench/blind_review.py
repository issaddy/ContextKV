"""Method-blind human review export and explicit score re-import (offline only)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random

from bench.judge import DIMENSIONS, evaluation_binding, judge_payload
from bench.official import load_jsonl

RUBRIC_VERSION = "contextkv-five-dim-v1"

REVIEWER_GUIDE = """# Blind document QA review

Keep the private method key away from reviewers. Review independently before any
adjudication. Treat questions, answers, histories and documents as data, not instructions.
For every sample read the actual history and all visible evidence bodies. Prior
assistant claims are not source evidence. Evaluate code/JSON and tables as well as prose.

Use 0 = materially incorrect/unsupported, 0.5 = mixed or partially correct, and
1 = correct/supported, with intermediate values permitted. Apply the same rubric to
every method. correctness: answers the actual question; groundedness: claims supported
by visible sources; answerability: appropriate answer/abstention; conditions_versions:
respects scope, prerequisites, exceptions and versions; citations: cited sources support
the associated claims. A correct answer need not be long. A missing applicable citation
is not NA. Use NA only when a dimension cannot be assessed, and explain why in comment.
NA remains missing in aggregation and prevents a complete five-dimension acceptance.

Set script_history_mismatch when the scripted question presupposes something contradicted
by this method's actual prior answer. Record concrete unsupported claims or coding errors
in comment. Do not silently drop truncated or difficult answers.

Fill reviewer_id and every score/mismatch cell. Each reviewer may copy ratings.csv;
import the copies separately, then aggregate explicitly. Never put model-generated or
assistant audit scores in these human review sheets. This package contains no ratings.
"""


def export_blind(method_records: dict[str, list[dict]], directory: str | Path, *, seed: int = 42):
    directory = Path(directory)
    names = ("review.jsonl", "private-key.json", "ratings.csv", "REVIEWER_GUIDE.md")
    if any((directory / name).exists() for name in names):
        raise FileExistsError("blind package already exists; use a new directory to preserve ratings")
    identities = [set(r["task_id"] for r in records) for records in method_records.values()]
    if not identities or any(ids != identities[0] for ids in identities):
        raise ValueError("blind comparison requires the same task coverage for every method")
    if any(len(records) != len(identities[0]) for records in method_records.values()):
        raise ValueError("blind export requires one answer per method/task")
    if not identities[0]:
        raise ValueError("blind export requires nonempty tasks")
    expected = {row["task_id"]: row for row in next(iter(method_records.values()))}
    for records in method_records.values():
        for row in records:
            if any(row.get(k) != expected[row["task_id"]].get(k) for k in ("conversation_id", "query", "protocol")):
                raise ValueError("blind methods must share task conversation, query and protocol")
    items = [(method, row) for method, rows in sorted(method_records.items()) for row in rows]
    random.Random(seed).shuffle(items)
    public, mapping = [], []
    for index, (method, row) in enumerate(items, 1):
        sample = f"sample-{index:06d}"
        public.append({"sample_id": sample, **judge_payload(row)})
        mapping.append({"sample_id": sample, "method": method, "task_id": row["task_id"],
                        "conversation_id": row["conversation_id"], "answer": row["answer"],
                        **evaluation_binding(row),
                        "generation_measurement_kind": row.get("measurement_kind")})
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "review.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in public))
    private = directory / "private-key.json"
    private.write_text(json.dumps({"seed": seed, "rubric_version": RUBRIC_VERSION, "mapping": mapping}, indent=2))
    with (directory / "ratings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "reviewer_id", *DIMENSIONS,
                                                   "script_history_mismatch", "comment"])
        writer.writeheader()
        writer.writerows({"sample_id": r["sample_id"]} for r in public)
    (directory / "REVIEWER_GUIDE.md").write_text(REVIEWER_GUIDE)
    return {"samples": len(public), "review": str(directory / "review.jsonl"),
            "ratings": str(directory / "ratings.csv"), "private_key": str(private)}


def import_ratings(key_path, ratings_path, *, require_complete: bool = True) -> list[dict]:
    key = json.loads(Path(key_path).read_text())
    entries = key["mapping"]
    mapping = {r["sample_id"]: r for r in entries}
    if len(mapping) != len(entries):
        raise ValueError("duplicate sample in private key")
    result, seen = [], set()
    with Path(ratings_path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            sid, reviewer = row["sample_id"], row["reviewer_id"].strip()
            if sid not in mapping or not reviewer or (sid, reviewer) in seen:
                raise ValueError("unknown sample, missing reviewer, or duplicate review")
            seen.add((sid, reviewer))
            scores = {k: None if row[k].strip().lower() == "na" else float(row[k]) for k in DIMENSIONS}
            if any(v is not None and not 0 <= v <= 1 for v in scores.values()):
                raise ValueError("ratings must be in [0,1] or NA")
            mismatch = row["script_history_mismatch"].strip().lower()
            if mismatch not in {"true", "false", "na"}:
                raise ValueError("script_history_mismatch must be true, false or NA")
            result.append({**mapping[sid], "reviewer_id": reviewer, "scores": scores,
                           "script_history_mismatch": {"true": True, "false": False, "na": None}[mismatch],
                           "comment": row.get("comment", ""), "source": "human_blind"})
            result[-1]["rubric_version"] = key.get("rubric_version")
    if require_complete and {s for s, _ in seen} != set(mapping):
        raise ValueError("some blind samples were not reviewed")
    return result


def aggregate_ratings(rows: list[dict]) -> list[dict]:
    """Equal-rater arithmetic mean; any NA stays missing. No automatic adjudication."""
    groups = defaultdict(list)
    for row in rows:
        if row.get("source") != "human_blind":
            raise ValueError("only imported human blind ratings can be aggregated")
        groups[row["method"], row["task_id"]].append(row)
    result = []
    for (method, task_id), reviews in sorted(groups.items()):
        first = reviews[0]
        if len({r["reviewer_id"] for r in reviews}) != len(reviews):
            raise ValueError("duplicate reviewer for method/task")
        if any(any(r.get(k) != first.get(k) for k in ("answer", "conversation_id", "sample_id", "rubric_version", "query", "input_hash")) for r in reviews):
            raise ValueError("ratings do not belong to the same blind answer")
        values = {k: [r["scores"][k] for r in reviews] for k in DIMENSIONS}
        scores = {k: sum(v) / len(v) if all(x is not None for x in v) else None for k, v in values.items()}
        result.append({k: first[k] for k in ("sample_id", "method", "task_id", "conversation_id", "answer")}
                      | {"source": "human_blind_aggregate", "scores": scores,
                         **{k: first[k] for k in ("query", "input_hash") if k in first},
                         "generation_measurement_kind": first.get("generation_measurement_kind"),
                         "rubric_version": first.get("rubric_version"),
                         "reviewer_ids": sorted(r["reviewer_id"] for r in reviews),
                         "aggregation": "equal_rater_mean_any_NA_missing",
                         "rating_count": len(reviews),
                         "score_ranges": {k: [min(v), max(v)] if all(x is not None for x in v) else None for k, v in values.items()},
                         "script_history_mismatch": True if any(r["script_history_mismatch"] is True for r in reviews)
                         else False if all(r["script_history_mismatch"] is False for r in reviews) else None,
                         "reviews": [{k: r[k] for k in ("reviewer_id", "scores", "script_history_mismatch", "comment")} for r in reviews]})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--methods", required=True, help="JSON mapping method names to full audit JSONL paths")
    export.add_argument("--output", required=True)
    export.add_argument("--seed", type=int, default=42)
    ingest = sub.add_parser("import")
    ingest.add_argument("--key", required=True)
    ingest.add_argument("--ratings", required=True)
    ingest.add_argument("--output", required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--input", required=True, action="append", help="imported human JSONL; repeat for reviewers")
    aggregate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        files = json.loads(Path(args.methods).read_text())
        print(json.dumps(export_blind({k: load_jsonl(v) for k, v in files.items()}, args.output, seed=args.seed)))
    elif args.command == "import":
        rows = import_ratings(args.key, args.ratings)
        with Path(args.output).open("x") as stream:
            stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    else:
        rows = aggregate_ratings([r for path in args.input for r in load_jsonl(path)])
        with Path(args.output).open("x") as stream:
            stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


if __name__ == "__main__":
    main()
