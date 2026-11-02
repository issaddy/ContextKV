"""Materialize public evaluation assets independently from training exports."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil


def version_catalog(rows):
    families = {}
    prepared = []
    for original in rows:
        row = dict(original)
        family = re.sub(r"v?\d+\.\d+\.\d+", "", row["document_id"])
        order = tuple(map(int, row["version"].split(".")))
        row.update(document_family=family, version_order=list(order))
        families[family] = max(order, families.get(family, order))
        prepared.append(row)
    for row in prepared:
        row["current"] = tuple(row["version_order"]) == families[row["document_family"]]
    return prepared


def prepare_public_assets(config):
    settings = config["assets"]["versionqa"]
    source = Path(settings["recorded_cache"]).resolve()
    target = Path(config["study"]["output_dir"])/"evaluation"/"versionqa"
    target.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (source/"versionqa-corpus.jsonl").read_text().splitlines()]
    prepared = version_catalog(rows)
    (target/"corpus.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in prepared))
    for name in ("tasks.jsonl", "references.jsonl", "split.json"):
        shutil.copyfile(source/("versionqa-"+name), target/name)
    for name in ("bm25s", "bge-m3"):
        destination = target/name
        if not destination.exists():
            destination.symlink_to(source/("versionqa-"+name), target_is_directory=True)
    manifest = {"dataset": "VersionQA", "questions": 100, "source_partition": "public_evaluation_only",
        "source_commit": "2a2cbe8f285557f99dc7a79d6df7134e1e3eccff", "recorded_cache": str(source),
        "passages": len(prepared), "documents": len({r["document_id"] for r in prepared}),
        "family_construction": "published filenames with semantic version component removed",
        "current_scope": "highest declared semantic version within each supplied document family",
        "unknown_dates": "publication and effective dates remain absent",
        "text_transformations": [], "training_use": False,
        "original_manifest": json.loads((source/"versionqa-corpus-manifest.json").read_text())}
    (target/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest
