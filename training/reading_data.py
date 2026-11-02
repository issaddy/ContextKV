"""Independent version and long-reading development tasks from training sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from training.task_data import read_jsonl, write_jsonl


def task_row(dataset, conversation, number, question, documents, answer, *, operation, scope):
    return {"dataset": {"name": dataset, "revision": "training-source-construction-v1"},
        "conversation_id": conversation, "task_id": f"{conversation}:{number}", "turn": number,
        "Collection": dataset+"/constructed_dev", "split": "dev",
        "input": [{"speaker": "user", "text": question}],
        "contexts": [{"document_id": doc["passage_id"], "text": doc["text"],
            "source": doc["source_uri"], "title": doc["title"],
            "version": doc["version"], "document_family": doc["document_family"],
            "valid_from": doc.get("valid_from"), "valid_to": doc.get("valid_to"),
            "published_at": doc.get("published_at"), "applicability": doc.get("applicability", {})} for doc in documents],
        "targets": [{"text": answer}], "coverage_mode": scope, "operation": operation,
        "source_document_ids": [doc["document_id"] for doc in documents],
        "label_origin": "constructed_from_native_training_sources"}


def build_reading_data(config):
    settings = config.get("training", {})
    sources_path = Path(settings.get("data_dir", "runs/study/data"))/"sources.jsonl"
    sources = sorted(read_jsonl(sources_path), key=lambda row: row["source_id"])
    root = Path(config.get("study", {}).get("output_dir", "runs/study"))/"evaluation"
    version_root, long_root = root/"version_dev", root/"long_dev"
    version_docs, version_tasks, version_refs = [], [], []
    families = settings.get("version_dev_families", 20)
    for index, original in enumerate(sources[:families]):
        family = f"training-dev-policy-{index:03}"
        program = f"Program {index+1}"
        documents, leaves = [], {}
        for version, start, end, offset in (("2019", "2019-01-01", "2024-01-01", 0),
                                            ("2024", "2024-01-01", None, 20)):
            for platform, base in (("Windows", 100), ("Linux", 80)):
                amount = base+offset+index
                fact = f"{program} policy version {version}: {platform} filing allowance is {amount} credits."
                body = original["text"]+"\n\n"+fact
                identifier = f"{family}-{version}-{platform.lower()}"
                document = {"passage_id": identifier, "document_id": identifier, "document_family": family,
                    "version": version, "valid_from": start, "valid_to": end,
                    "published_at": start, "applicability": {"PLATFORM": platform},
                    "title": f"{program} {version} {platform} allowance", "text": body,
                    "source_uri": f"study-training://version_dev/{identifier}",
                    "derived_from": {"source_id": original["source_id"], "start": 0, "end": len(original["text"])}}
                documents.append(document)
                leaves[version, platform] = {"span_id": f"{identifier}:allowance", "source_id": identifier,
                    "start": len(original["text"])+2, "end": len(body), "quote": fact,
                    "value": amount, "unit": "credits", "platform": platform, "version": version}
        version_docs.extend(documents)
        cases = [
            (f"What is the current filing allowance for {program} on Windows?", "lookup", [("2024", "Windows")]),
            ("Change the platform to Linux. What is its allowance?", "lookup", [("2024", "Linux")]),
            ("What was its Linux allowance under policy version 2019?", "lookup", [("2019", "Linux")]),
            ("Compare the Linux allowances in policy versions 2019 and 2024.", "compare", [("2019", "Linux"), ("2024", "Linux")])]
        for number, (question, operation, keys) in enumerate(cases, 1):
            selected = [leaves[key] for key in keys]
            answer = "; ".join(f"{leaf['version']} {leaf['platform']}: {leaf['value']} credits" for leaf in selected)
            row = task_row("version_dev", family, number, question, documents, answer,
                           operation=operation, scope="selected_sources")
            version_tasks.append(row)
            version_refs.append({"task_id": row["task_id"], "split": "dev", "document_family": family,
                "answer": answer, "source_spans": selected, "operation": operation,
                "expected_values": [leaf["value"] for leaf in selected],
                "expected_versions": [leaf["version"] for leaf in selected],
                "expected_platform": "Windows" if number == 1 else "Linux"})
    long_docs, long_tasks, long_refs, traces = [], [], [], []
    long_cases = settings.get("long_dev_cases", 20)
    minimum_chars = settings.get("long_dev_min_characters", 200000)
    for index in range(long_cases):
        conversation = f"training-dev-long-{index:03}"
        documents, leaves, total_chars = [], [], 0
        cursor = (index*17+families) % len(sources)
        while total_chars < minimum_chars or len(documents) < 3:
            original = sources[cursor % len(sources)]
            cursor += 1
            number = len(documents)+1
            item = f"Inventory-{index+1}-{number:03}"
            fact = f"Inventory row: item {item}; amount {number} credits; category {'A' if number % 2 else 'B'}."
            body = original["text"]+"\n\n"+fact
            identifier = f"{conversation}-source-{number:03}"
            document = {"passage_id": identifier, "document_id": identifier, "version": "constructed-dev-v1",
                "document_family": conversation, "title": f"Inventory document {number}", "text": body,
                "source_uri": f"study-training://long_dev/{identifier}",
                "derived_from": {"source_id": original["source_id"], "start": 0, "end": len(original["text"])}}
            documents.append(document)
            leaves.append({"claim_id": item, "source_id": identifier, "start": len(original["text"])+2,
                "end": len(body), "quote": fact, "entity": item, "value": number, "unit": "credits"})
            total_chars += len(body)
        count = len(leaves)
        total = sum(leaf["value"] for leaf in leaves)
        question = ("Read every supplied document. Count all unique items explicitly listed in the Inventory rows "
                    "and sum their amounts in credits. Report both the item count and the total credits.")
        answer = f"{count} unique inventory items; {total} credits in total."
        row = task_row("long_dev", conversation, 1, question, documents, answer,
                       operation="count+sum", scope="all_supplied_documents")
        long_docs.extend(documents)
        long_tasks.append(row)
        graph = [{"claim_id": "inventory-count", "kind": "derived", "operation": "count",
                  "parent_claim_ids": [leaf["claim_id"] for leaf in leaves], "expected_value": count},
                 {"claim_id": "inventory-sum", "kind": "derived", "operation": "sum",
                  "parent_claim_ids": [leaf["claim_id"] for leaf in leaves], "expected_value": total, "unit": "credits"}]
        long_refs.append({"task_id": row["task_id"], "split": "dev", "answer": answer,
            "expected_count": count, "expected_sum": total, "source_spans": leaves,
            "source_characters": total_chars, "operation_graph": graph})
        traces.append({"task_id": row["task_id"], "split": "dev", "source_spans": leaves,
            "operation_graph": graph, "status": "construction_supervision_pending_actual_READ_rollout"})
    for destination, documents, tasks, references in ((version_root, version_docs, version_tasks, version_refs),
                                                     (long_root, long_docs, long_tasks, long_refs)):
        destination.mkdir(parents=True, exist_ok=True)
        write_jsonl(destination/"corpus.jsonl", documents)
        write_jsonl(destination/"tasks.jsonl", tasks)
        write_jsonl(destination/"references.jsonl", references)
    write_jsonl(long_root/"reading_trace_supervision.jsonl", traces)
    hop_root = root/"multihop_dev"
    hop_docs, hop_tasks, hop_refs = [], [], []
    for index in range(settings.get("multihop_dev_cases", 20)):
        identifier = f"training-dev-hop-{index:03}"
        department, organization, foundation = f"Department {index+1}", f"Organization {index+1}", f"Foundation {index+1}"
        year = 1980+index
        facts = (f"{department} belongs to {organization}.", f"{organization} is funded by {foundation}.",
                 f"{foundation} was established in {year}.")
        documents, spans = [], []
        for hop, fact in enumerate(facts):
            original = sources[(families+long_cases*17+index*3+hop) % len(sources)]
            body = original["text"]+"\n\n"+fact
            source_id = f"{identifier}-hop-{hop+1}"
            documents.append({"passage_id": source_id, "document_id": source_id, "document_family": identifier,
                "version": "constructed-dev-v1", "title": f"{identifier} relation {hop+1}", "text": body,
                "source_uri": f"study-training://multihop_dev/{source_id}",
                "derived_from": {"source_id": original["source_id"], "start": 0, "end": len(original["text"])}})
            spans.append({"claim_id": f"{identifier}:claim:{hop+1}", "source_id": source_id,
                "start": len(original["text"])+2, "end": len(body), "quote": fact})
        question = f"In which year was the foundation funding {department}'s parent organization established?"
        row = task_row("multihop_dev", identifier, 1, question, documents, str(year),
                       operation="infer", scope="selected_sources")
        hop_tasks.append(row)
        hop_docs.extend(documents)
        hop_refs.append({"task_id": row["task_id"], "split": "dev", "answer": str(year), "source_spans": spans,
            "operation_graph": [{"claim_id": f"{identifier}:answer", "kind": "derived", "operation": "infer",
                                 "parent_claim_ids": [span["claim_id"] for span in spans], "expected_value": year}],
            "open_need_bindings": [{"bound_entities": [organization], "missing_relation": "funded_by",
                                    "parent_claim_ids": [spans[0]["claim_id"]]},
                                   {"bound_entities": [foundation], "missing_relation": "established_in",
                                    "parent_claim_ids": [spans[1]["claim_id"]]}]})
    write_jsonl(hop_root/"corpus.jsonl", hop_docs)
    write_jsonl(hop_root/"tasks.jsonl", hop_tasks)
    write_jsonl(hop_root/"references.jsonl", hop_refs)
    manifest = {"schema_version": 1, "split": "dev", "source_partition": "native_multidoc2dial_train",
        "source_path": str(sources_path.resolve()), "construction": "literal training documents plus controlled original dev facts",
        "version": {"families": families, "documents": len(version_docs), "turns": len(version_tasks),
                    "path": str((version_root/"tasks.jsonl").resolve())},
        "long": {"cases": len(long_tasks), "documents": len(long_docs), "minimum_characters": minimum_chars,
                 "path": str((long_root/"tasks.jsonl").resolve()), "token_counts": "measured by configured reader during execution"},
        "multihop": {"cases": len(hop_tasks), "documents": len(hop_docs),
                     "path": str((hop_root/"tasks.jsonl").resolve())},
        "public_evaluation_assets_read": []}
    root.mkdir(parents=True, exist_ok=True)
    (root/"development_construction_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.yaml")
    args = parser.parse_args(argv)
    print(json.dumps(build_reading_data(yaml.safe_load(Path(args.config).read_text())), indent=2))


if __name__ == "__main__":
    main()
