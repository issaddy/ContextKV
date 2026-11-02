from dataclasses import replace
import json

import pytest
import yaml

from controller.schema import OnlineContext
from training.task_data import read_jsonl
from training.task_rollout import run_actual_rollouts


@pytest.mark.asyncio
async def test_rollout_uses_committed_generated_history_and_resumes_recorded_turns(tmp_path, monkeypatch):
    from training import task_rollout
    root = tmp_path/"data"
    root.mkdir()
    rows = [{"task_id": f"c:{i}", "conversation_id": "c", "turn": i, "question": f"Question {i}?",
             "grounding": [], "reference_answer": "AUTHOR_REFERENCE_SHOULD_NOT_APPEAR"} for i in (1, 2)]
    (root/"original_train_turns.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    corpus_path = tmp_path/"corpus.jsonl"
    corpus_path.write_text(json.dumps({"_id": "p", "text": "Original source body.", "version": "v1"})+"\n")
    reader_path = tmp_path/"reader.yaml"
    reader_path.write_text(yaml.safe_dump({"engine": {"kind": "mock"},
        "planner": {"strategy": "always_rebuild"}, "context": {"max_total_model_tokens": 8192,
        "max_new_tokens": 128}}))
    retriever_path = tmp_path/"retriever.yaml"
    retriever_path.write_text("{}\n")
    class Retriever:
        corpus_snapshot_id = "v1"
        manifest = {"kind": "fixture"}
        def retrieve_turn(self, turn):
            return replace(turn, contexts=(OnlineContext("p", "Original source body."),))
    monkeypatch.setattr(task_rollout, "create_retriever", lambda *args, **kwargs: Retriever())
    config = {"training": {"data_dir": str(root), "rollout_state_dir": str(tmp_path/"state"),
        "reader_config": str(reader_path), "retriever_config": str(retriever_path),
        "evidence_corpus": str(corpus_path)}}
    manifest = await run_actual_rollouts(config, min_conversations=1, min_turns=2, concurrency=1)
    exported = list(read_jsonl(root/"turns.jsonl"))
    assert manifest["generated_turns"] == 2
    assert exported[0]["actual_history"] == []
    assert exported[1]["actual_history"][0]["assistant_text"] == exported[0]["answer"]
    assert exported[1]["actual_history"][0]["assistant_text"] != rows[0]["reference_answer"]
    assert exported[1]["response"]["ledger_version"] == 2
    resumed = await run_actual_rollouts(config, min_conversations=1, min_turns=2, concurrency=1)
    assert resumed["generated_turns"] == 0 and len(list(read_jsonl(root/"turns.jsonl"))) == 2
