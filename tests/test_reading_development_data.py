import json

from training.reading_data import build_reading_data
from training.task_data import read_jsonl


def test_development_cases_use_training_sources_and_exact_operation_leaves(tmp_path):
    data = tmp_path/"data"
    data.mkdir()
    sources = [{"source_id": f"train-{i}", "text": f"Original training text {i}. "*5,
                "title": f"Training {i}", "version": "train"} for i in range(4)]
    (data/"sources.jsonl").write_text("".join(json.dumps(row)+"\n" for row in sources))
    output = tmp_path/"study"
    manifest = build_reading_data({"study": {"output_dir": str(output)}, "training": {
        "data_dir": str(data), "version_dev_families": 2, "long_dev_cases": 2,
        "long_dev_min_characters": 200, "multihop_dev_cases": 2}})
    assert manifest["public_evaluation_assets_read"] == []
    assert manifest["version"]["families"] == 2 and manifest["version"]["turns"] == 8
    for group in ("version_dev", "long_dev", "multihop_dev"):
        documents = {row["passage_id"]: row for row in read_jsonl(output/"evaluation"/group/"corpus.jsonl")}
        references = list(read_jsonl(output/"evaluation"/group/"references.jsonl"))
        for document in documents.values():
            origin = next(row for row in sources if row["source_id"] == document["derived_from"]["source_id"])
            assert document["text"].startswith(origin["text"])
        for row in references:
            for leaf in row["source_spans"]:
                assert documents[leaf["source_id"]]["text"][leaf["start"]:leaf["end"]] == leaf["quote"]
            if group == "long_dev":
                assert row["expected_count"] == len(row["source_spans"])
                assert row["expected_sum"] == sum(leaf["value"] for leaf in row["source_spans"])
            if group == "multihop_dev":
                assert len(row["open_need_bindings"]) == 2
                assert len(row["operation_graph"][0]["parent_claim_ids"]) == 3
