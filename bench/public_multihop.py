"""Container-side official 2WikiMultihopQA source, frozen cohorts and scoring."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

from training.reader_data import file_digest, rows, write_rows


AUTHOR_REVISION = "612bc5039a457880d9e7d84c3b0a4cf154b70e4f"
AUTHOR_MIRROR = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA"
OFFICIAL_REPOSITORY = "https://github.com/Alab-NII/2wikimultihop"


def download(output):
    import httpx
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    sources = {"dev.parquet": f"{AUTHOR_MIRROR}/resolve/{AUTHOR_REVISION}/dev.parquet",
        "README.author.md": f"{AUTHOR_MIRROR}/resolve/{AUTHOR_REVISION}/README.md",
        "evaluate.original.py": "https://raw.githubusercontent.com/Alab-NII/2wikimultihop/main/2wikimultihop_evaluate.py",
        "LICENSE": "https://raw.githubusercontent.com/Alab-NII/2wikimultihop/main/LICENSE"}
    files = {}
    started = time.monotonic()
    with httpx.Client(follow_redirects=True, timeout=300, trust_env=True) as client:
        for name, url in sources.items():
            path = root/name
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with path.open("wb") as stream:
                    for chunk in response.iter_bytes():
                        stream.write(chunk)
            files[name] = {"source_url": url, "sha256": file_digest(path), "bytes": path.stat().st_size}
    manifest = {"dataset": "2WikiMultihopQA", "repository": OFFICIAL_REPOSITORY,
        "author_mirror": AUTHOR_MIRROR, "author_revision": AUTHOR_REVISION, "source_split": "official labeled dev",
        "files": files, "download_wall_ms": (time.monotonic()-started)*1000,
        "proxy_policy": "existing environment proxy via httpx trust_env; credentials never recorded"}
    (root/"source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def convert(source_dir, output, *, dev_count=40, test_count=40, seed=42):
    import pyarrow.parquet as parquet
    source_dir, output = Path(source_dir), Path(output)
    population = parquet.read_table(source_dir/"dev.parquet").to_pylist()
    ordered = sorted(population, key=lambda row: hashlib.sha256(f"{seed}:{row['_id']}".encode()).hexdigest())
    selected = {"dev": ordered[:dev_count], "test": ordered[dev_count:dev_count+test_count]}
    if len(selected["dev"]) != dev_count or len(selected["test"]) != test_count:
        raise ValueError("declared 2Wiki cohort is larger than its public source population")
    corpus, cohorts, gold = {}, {}, {}
    for split, chosen in selected.items():
        tasks, gold[split] = [], []
        for item in chosen:
            context = json.loads(item["context"])
            support = json.loads(item["supporting_facts"])
            evidence = json.loads(item["evidences"])
            contexts, sentence_spans = [], []
            for index, (title, sentences) in enumerate(context):
                document = f"2wiki:{item['_id']}:{index}"
                text = "\n".join(sentences)
                cursor = 0
                for sid, sentence in enumerate(sentences):
                    sentence_spans.append({"source_id": document, "title": title, "sentence_id": sid,
                        "start": cursor, "end": cursor+len(sentence), "quote": sentence})
                    cursor += len(sentence)+1
                source = {"document_id": document, "text": text, "title": title, "version": AUTHOR_REVISION,
                    "source": f"2wikimultihopqa://{item['_id']}/{index}", "document_family": title}
                contexts.append(source)
                corpus[document] = {"_id": document, **source}
            identifier = "2wiki:"+item["_id"]
            reference = {"answer": item["answer"], "supporting_facts": support, "evidences": evidence,
                "sentence_spans": sentence_spans, "type": item["type"], "original_id": item["_id"]}
            tasks.append({"task_id": identifier, "conversation_id": identifier, "turn": 1,
                "Collection": "2wikimultihopqa/official-dev", "split": split,
                "dataset": {"name": "2WikiMultihopQA", "revision": AUTHOR_REVISION},
                "input": [{"speaker": "user", "text": item["question"]}], "contexts": contexts,
                "coverage_mode": "selected_sources", "targets": [{"text": item["answer"]}],
                "offline_reference": reference})
            gold[split].append({**item, "context": context, "supporting_facts": support, "evidences": evidence})
        write_rows(output/split/"tasks.jsonl", tasks)
        (output/split/"gold.json").write_text(json.dumps(gold[split], ensure_ascii=False)+"\n")
        cohorts[split] = {"task_ids": [row["task_id"] for row in tasks],
            "types": dict(Counter(row["offline_reference"]["type"] for row in tasks)),
            "tasks_sha256": file_digest(output/split/"tasks.jsonl"), "gold_sha256": file_digest(output/split/"gold.json")}
    write_rows(output/"passages.jsonl", corpus.values())
    manifest = {"dataset": "2WikiMultihopQA", "author_revision": AUTHOR_REVISION,
        "source_parquet_sha256": file_digest(source_dir/"dev.parquet"), "population": len(population),
        "sampling": "sha256(seed:original_id) ascending; disjoint consecutive dev/test slices", "seed": seed,
        "split_kind": "project_disjoint_cohorts_from_official_labeled_dev; not official hidden test",
        "cohorts": cohorts, "context_policy": "all original context paragraphs, including distractors; no gold source filtering",
        "sentence_serialization": "original sentences joined by newline; exact Unicode sentence spans retained offline",
        "training_use": False, "public_dataset_coverage": (dev_count+test_count)/len(population),
        "corpus_sha256": file_digest(output/"passages.jsonl"), "source_manifest": str(source_dir/"source_manifest.json")}
    (output/"cohort_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    spec = {"inputs": {split: str(output/split/"tasks.jsonl") for split in ("dev", "test")},
        "corpus_id": "twowiki", "corpus_path": str(output/"passages.jsonl"),
        "retriever_config": str(output/"retriever.yaml"), "supplied_documents": True,
        "coverage_mode": "selected_sources", "protocol": "P3", "split_kind": manifest["split_kind"],
        "primary_metric": "2wiki_answer_f1", "normalization": 100.0, "weight": 1,
        "scoring": {"implementation": "official original 2wikimultihop_evaluate.py", "source_dir": str(source_dir)}}
    (output/"study_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    recipe = yaml.safe_load(Path("configs/study_multidoc_retriever.yaml").read_text())
    recipe.update(sparse_index=str(output/"bm25"), dense_index=str(output/"dense"),
                  query_adapter_path=None, query_mode="current")
    (output/"retriever.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    return manifest


async def build_indexes(output, *, device="cuda:1", physical_device=1):
    import os
    from bench.study_resources import StudyResourceSampler, integrate_physical
    root = Path(output)
    sampler = StudyResourceSampler([physical_device], [os.getpid()])
    await sampler.start()
    started = time.monotonic()
    measured = {}
    def operation():
        from adapters.hybrid_retrieval import DenseIndex, LocalBGE
        from adapters.retrieval import BM25SRetriever, load_beir_corpus
        from controller.schema import OnlineContext
        corpus = load_beir_corpus(root/"passages.jsonl")
        snapshot = "2wiki-"+file_digest(root/"passages.jsonl")[:20]
        sparse = BM25SRetriever(corpus, corpus_snapshot_id=snapshot, cpu_threads=1)
        sparse.save(root/"bm25")
        encoder = LocalBGE("models/bge-m3", device=device, dtype="bfloat16", batch_size=16, max_length=8192)
        try:
            dense = DenseIndex.build([OnlineContext(**row) for row in corpus.values()], encoder,
                root/"dense", corpus_snapshot_id=snapshot, show_progress=False)
            measured.update(encoder_usage=encoder.usage, dense=dense.manifest, sparse=sparse.manifest)
        finally:
            encoder.close()
    try:
        await asyncio.to_thread(operation)
    finally:
        ended = time.monotonic()
        await sampler.stop()
        (root/"index_resource_samples.json").write_text(json.dumps(sampler.observation(), indent=2)+"\n")
        (root/"index_cost.json").write_text(json.dumps({"window": [started, ended],
            "physical": integrate_physical(sampler.samples, started, ended), "observed": measured,
            "scope": "offline source indexing; no training or gold-label use"}, indent=2)+"\n")
    return {"dataset": "2WikiMultihopQA", "index": "complete", "passages": measured["dense"]["passages"]}


def score(source_dir, tasks_path, generation_dir):
    root = Path(generation_dir)
    tasks = {row["task_id"]: row for row in rows(tasks_path)}
    results = rows(root/"results.unscored.jsonl")
    output = root/"scoring"/"2wiki"
    output.mkdir(parents=True, exist_ok=True)
    predictions, gold, individual = {"answer": {}, "sp": {}, "evidence": {}}, [], {}
    import runpy
    evaluator = runpy.run_path(str(Path(source_dir)/"evaluate.original.py"))
    for row in results:
        if row["status"] != "completed":
            continue
        reference = tasks[row["task_id"]]["offline_reference"]
        original = reference["original_id"]
        answer = re.sub(r"\[E_[A-Za-z0-9_]+\]", "", row["answer"]).strip()
        outcome = row.get("reading_outcome") or row.get("attempt", {}).get("reading_outcome")
        claims = outcome["claims"] if outcome is not None else []
        spans = [span for claim in claims for span in claim["source_spans"]]
        predicted_sp = sorted({(sentence["title"], sentence["sentence_id"])
            for sentence in reference["sentence_spans"] for span in spans
            if span["source_key"]["document_id"] == sentence["source_id"]
            and span["start"] < sentence["end"] and sentence["start"] < span["end"]})
        predicted_evidence = [[claim["entity"], claim["relation"], str(claim["value"])] for claim in claims
                              if claim["kind"] != "derived" and claim["entity"] and claim["relation"] and claim["value"] is not None]
        predictions["answer"][original], predictions["sp"][original], predictions["evidence"][original] = answer, predicted_sp, predicted_evidence
        gold.append({"_id": original, **{name: reference[name] for name in ("answer", "supporting_facts", "evidences")}})
        individual[row["task_id"]] = {"2wiki_answer_em": float(evaluator["exact_match_score"](answer, reference["answer"]))*100,
            "2wiki_answer_f1": evaluator["f1_score"](answer, reference["answer"])[0]*100}
    prediction_path, gold_path = output/"predictions.json", output/"gold.completed.json"
    prediction_path.write_text(json.dumps(predictions, ensure_ascii=False)+"\n")
    gold_path.write_text(json.dumps(gold, ensure_ascii=False)+"\n")
    started = time.monotonic()
    completed = subprocess.run([sys.executable, str(Path(source_dir)/"evaluate.original.py"),
        str(prediction_path), str(gold_path)], capture_output=True, text=True, check=True)
    (output/"official.stdout.json").write_text(completed.stdout)
    scored = [{**row, "metrics": {**row.get("metrics", {}), **individual[row["task_id"]]},
        "scoring": {"status": "scored", "source": "official 2Wiki answer normalization/EM/F1", "cost_scope": "offline CPU scoring"}}
        for row in results if row["task_id"] in individual]
    write_rows(root/"results.scored.jsonl", scored)
    manifest = {"metric": "official 2Wiki original answer/sp/evidence/joint metrics", "scored": len(scored),
        "planned": len(tasks), "missing_task_ids": sorted(set(tasks)-set(individual)),
        "answer_projection": "remove only E_ citation markup; no semantic extraction, answer repair or gold-guided edit",
        "support_projection": "original sentence overlap of actual claim source spans; no gold sentence selection",
        "raw_answers": str(root/"results.unscored.jsonl"), "official": json.loads(completed.stdout),
        "source_script_sha256": file_digest(Path(source_dir)/"evaluate.original.py"),
        "wall_ms": (time.monotonic()-started)*1000, "model_calls": 0,
        "scope": "observed completed subset; missing/failed cases are not assigned a fabricated quality value"}
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "convert", "index", "score"))
    parser.add_argument("--source-dir", default="data/2wikimultihopqa/source")
    parser.add_argument("--output", default="data/2wikimultihopqa/cohort40")
    parser.add_argument("--dev-count", type=int, default=40)
    parser.add_argument("--test-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks")
    parser.add_argument("--generation-dir")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--physical-device", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "download":
        result = download(args.source_dir)
    elif args.command == "convert":
        result = convert(args.source_dir, args.output, dev_count=args.dev_count, test_count=args.test_count, seed=args.seed)
    elif args.command == "index":
        result = asyncio.run(build_indexes(args.output, device=args.device, physical_device=args.physical_device))
    else:
        result = score(args.source_dir, args.tasks, args.generation_dir)
    print(json.dumps({key: value for key, value in result.items() if key in {"dataset", "population", "scored", "planned", "sampling"}}))


if __name__ == "__main__":
    main()
