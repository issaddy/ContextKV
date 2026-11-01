import json
from pathlib import Path

from adapters.mtrag import OnlineSession, load_rows
from bench.quality_stress import materialize


def test_stress_workload_keeps_labels_offline_and_uses_generated_history(tmp_path):
    source = Path(__file__).parents[1] / "bench/assets/conversation_quality_stress.json"
    output = tmp_path / "workload"
    materialize(source, output)
    tasks = load_rows(output / "tasks.jsonl")
    split = json.loads((output / "split.json").read_text())
    assert set(split["dev"]).isdisjoint(split["test"])
    assert all("targets" not in row and "support" not in row for row in tasks)
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    narrowing = [row for row in labels if "scope_narrowing" in row["tags"]]
    assert len(narrowing) == 4
    assert all(row["task_id"].endswith("<::>2") and row["support"] for row in narrowing)

    session = OnlineSession([row for row in tasks if row["conversation_id"] == split["dev"][0]], "P3")
    first = session.current()
    assert first.history == () and first.contexts == ()
    session.commit(first.task_id, "An actual generated answer.")
    second = session.current()
    assert second.history[-1].text == "An actual generated answer."
    assert second.history[-1].speaker == "assistant"
    assert (output / "references.jsonl").is_file()
    assert (output / "qrels.tsv").is_file()
