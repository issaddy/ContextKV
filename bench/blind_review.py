"""Method-blind human review export and explicit score re-import (offline only)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

from bench.judge import DIMENSIONS, judge_payload
from bench.official import load_jsonl


def export_blind(method_records: dict[str, list[dict]], directory: str | Path, *, seed: int = 42):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identities = [set(r["task_id"] for r in records) for records in method_records.values()]
    if not identities or any(ids != identities[0] for ids in identities):
        raise ValueError("blind comparison requires the same task coverage for every method")
    if any(len(records) != len(identities[0]) for records in method_records.values()):
        raise ValueError("blind export requires one answer per method/task")
    items = [(method, row) for method, rows in sorted(method_records.items()) for row in rows]
    random.Random(seed).shuffle(items)
    public, mapping = [], []
    for index, (method, row) in enumerate(items, 1):
        sample = f"sample-{index:06d}"
        public.append({"sample_id": sample, **judge_payload(row)})
        mapping.append({"sample_id": sample, "method": method, "task_id": row["task_id"],
                        "conversation_id": row["conversation_id"], "answer": row["answer"]})
    (directory / "review.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in public))
    private = directory / "private-key.json"
    private.write_text(json.dumps({"seed": seed, "mapping": mapping}, indent=2))
    with (directory / "ratings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "reviewer_id", *DIMENSIONS,
                                                   "script_history_mismatch", "comment"])
        writer.writeheader()
        writer.writerows({"sample_id": r["sample_id"]} for r in public)
    return {"samples": len(public), "review": str(directory / "review.jsonl"),
            "ratings": str(directory / "ratings.csv"), "private_key": str(private)}


def import_ratings(key_path, ratings_path, *, require_complete: bool = True) -> list[dict]:
    mapping = {r["sample_id"]: r for r in json.loads(Path(key_path).read_text())["mapping"]}
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
    if require_complete and {s for s, _ in seen} != set(mapping):
        raise ValueError("some blind samples were not reviewed")
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
    args = parser.parse_args(argv)
    if args.command == "export":
        files = json.loads(Path(args.methods).read_text())
        print(json.dumps(export_blind({k: load_jsonl(v) for k, v in files.items()}, args.output, seed=args.seed)))
    else:
        rows = import_ratings(args.key, args.ratings)
        Path(args.output).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    main()
