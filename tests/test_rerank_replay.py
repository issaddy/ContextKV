from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.mtrag import OnlineContext
from adapters.qwen3_reranker import (DEFAULT_INSTRUCTION, PREFIX, SUFFIX, LocalQwen3Reranker,
                                    format_instruction, prepare_inputs, yes_no_probabilities)
from bench import rerank_replay
from bench.retrieval_eval import file_source


class Tensor(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def to(self, device):
        return self

    def float(self):
        return Tensor(self.astype(np.float32))

    def exp(self):
        return Tensor(np.exp(self))


def log_softmax(value, dim):
    shifted = value-np.max(value, axis=dim, keepdims=True)
    return Tensor(shifted-np.log(np.exp(shifted).sum(axis=dim, keepdims=True)))


TORCH_FIXTURE = SimpleNamespace(stack=lambda values, dim: Tensor(np.stack(values, axis=dim)),
    nn=SimpleNamespace(functional=SimpleNamespace(log_softmax=log_softmax)),
    inference_mode=nullcontext, float32="float32", bfloat16="bfloat16")


class TokenizerFixture:
    padding_side = "left"
    truncation_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [3, 4] if text == PREFIX else [5, 6] if text == SUFFIX else [ord(c)+10 for c in text]

    def convert_tokens_to_ids(self, text):
        return {"no": 1, "yes": 2}[text]

    def __call__(self, texts, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [self.encode(text) for text in texts]}

    def pad(self, inputs, **kwargs):
        assert self.padding_side == "left"
        rows = inputs["input_ids"]
        width = max(map(len, rows))
        return {"input_ids": Tensor([[0]*(width-len(row))+row for row in rows]),
                "attention_mask": Tensor([[0]*(width-len(row))+[1]*len(row) for row in rows])}


def test_official_prompt_format_and_complete_wrapper_survive_truncation():
    assert format_instruction("q", "doc", None) == f"<Instruct>: {DEFAULT_INSTRUCTION}\n<Query>: q\n<Document>: doc"
    tokenizer = TokenizerFixture()
    inputs, lengths = prepare_inputs(tokenizer, ["a", "abcdefghij"], [3, 4], [5, 6], 8)
    assert inputs["input_ids"].tolist() == [[0, 0, 0, 3, 4, 107, 5, 6], [3, 4, 107, 108, 109, 110, 5, 6]]
    assert inputs["attention_mask"].tolist()[0] == [0, 0, 0, 1, 1, 1, 1, 1]
    assert lengths == [{"original_input_tokens": 5, "used_input_tokens": 5, "truncated": False},
                       {"original_input_tokens": 14, "used_input_tokens": 8, "truncated": True}]
    with pytest.raises(ValueError, match="leave room"):
        prepare_inputs(tokenizer, ["a"], [3, 4], [5, 6], 4)


def test_yes_no_uses_last_position_only_and_normalizes_correct_order():
    logits = np.zeros((2, 3, 4))
    logits[:, 0, 1] = 1000  # Earlier positions and unrelated vocabulary must not decide the score.
    logits[:, -1, 3] = 1000
    logits[0, -1, 2] = np.log(3)
    logits[1, -1, 1] = np.log(3)
    assert yes_no_probabilities(Tensor(logits), 1, 2, TORCH_FIXTURE) == pytest.approx([.75, .25])


def test_local_loader_uses_causal_model_offline_and_never_generates(monkeypatch):
    loads, forwards = [], []
    tokenizer = TokenizerFixture()
    class Model:
        device = SimpleNamespace(type="cpu")
        dtype = "float32"
        def to(self, **kwargs):
            return self
        def eval(self):
            return self
        def __call__(self, **kwargs):
            forwards.append(kwargs)
            assert kwargs["logits_to_keep"] == 1 and kwargs["use_cache"] is False
            logits = np.zeros((len(kwargs["input_ids"]), 1, 4))
            logits[:, -1, 2] = np.log(3)
            return SimpleNamespace(logits=Tensor(logits))
    def load_tokenizer(path, **kwargs):
        loads.append(("tokenizer", path, kwargs))
        return tokenizer
    def load_model(path, **kwargs):
        loads.append(("causal_model", path, kwargs))
        return Model()
    monkeypatch.setitem(sys.modules, "torch", TORCH_FIXTURE)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model)))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fixture")
    scorer = LocalQwen3Reranker("/local/snapshot", batch_size=2, max_length=128)
    rows = [OnlineContext(str(i), "untouched long document "*20, title="Title") for i in range(3)]
    assert scorer.score("query", rows) == pytest.approx([.75]*3)
    assert len(forwards) == 2 and len(scorer.last_pairs) == 3
    assert all(pair["truncated"] and pair["used_input_tokens"] == 128 for pair in scorer.last_pairs)
    assert scorer.truncated_pairs == scorer.texts_processed == 3
    assert all(call[2]["local_files_only"] is True for call in loads)
    assert loads[0][2]["padding_side"] == "left" and loads[0][2]["truncation_side"] == "right"
    scorer.close()
    assert not hasattr(scorer, "model") and scorer.manifest["max_length"] == 128


def source_data():
    tasks = [{"task_id": f"s::{i}", "conversation_id": "s", "turn": i,
              "user_questions": [f"question {j}" for j in range(1, i+1)]} for i in (1, 2)]
    contexts = [OnlineContext(f"d{i}", f"original text {i}", title=f"title {i}") for i in range(5)]
    rows = []
    for task in tasks:
        base = {**{k: task[k] for k in ("task_id", "conversation_id", "turn")}, "query_mode": "current",
            "query_text": task["user_questions"][-1], "status": "complete",
            "candidate_scores": [1.]*5, "contexts": [asdict(c) for c in contexts]}
        ids = [c.document_id for c in contexts]
        rows.extend([{**base, "stage": "hybrid", "method": "current/hybrid", "candidate_ids": ids,
                      "rerank_candidate_ids": None, "elapsed_ms": 10},
                     {**base, "stage": "hybrid_rerank", "method": "current/hybrid_rerank", "candidate_ids": ids[::-1],
                      "rerank_candidate_ids": ids, "elapsed_ms": 15}])
    return rows, {"status": "complete", "tasks": tasks, "reranker": {"max_length": 1024},
                  "sparse": {"corpus_sha256": "fixture"}}, contexts


def test_frozen_source_checks_cohort_query_and_original_rerank_pool():
    rows, manifest, _ = source_data()
    assert len(rerank_replay.frozen_pools(rows, manifest, pool_k=5)) == 2
    for mutate, message in ((lambda r: r.pop(), "cohort"),
        (lambda r: r[0].update(query_text="future answer"), "query/task"),
        (lambda r: r[1].update(rerank_candidate_ids=["different"]), "candidate pool")):
        changed = deepcopy(rows)
        mutate(changed)
        with pytest.raises(ValueError, match=message):
            rerank_replay.frozen_pools(changed, manifest, pool_k=5)


class ScorerFixture:
    manifest = {"measurement_kind": "mock_contract", "max_length": 1024}
    def __init__(self, fail=False):
        self.calls, self.last_pairs = [], []
        self.texts_processed = self.truncated_pairs = 0
        self.closed = False
        self.fail = fail
    def score(self, query, contexts):
        self.calls.append((query, contexts))
        if self.fail and query == "question 2":
            raise RuntimeError("fixture model failure")
        self.texts_processed += len(contexts)
        self.last_pairs = [{"document_id": c.document_id, "truncated": False} for c in contexts]
        return [1.]*len(contexts)
    def close(self):
        self.closed = True


def test_replay_preserves_source_and_never_exposes_reference_scores_to_model():
    rows, manifest, contexts = source_data()
    original = deepcopy(rows)
    pools = rerank_replay.frozen_pools(rows, manifest, pool_k=5)
    scorer = ScorerFixture(fail=True)
    results = list(rerank_replay.replay_pools(pools, {c.document_id: c for c in contexts}, scorer))
    assert rows == original
    assert [c.document_id for c in scorer.calls[0][1]] == [f"d{i}" for i in range(5)]
    assert all(c.score is None for c in scorer.calls[0][1])
    assert results[0]["candidate_ids"] == [f"d{i}" for i in range(5)]  # Deterministic equal-score tie.
    assert results[1]["status"] == "failed" and results[1]["candidate_ids"] == []
    summary = rerank_replay.analyze_replay(pools, results, {"s::1": {"d0": 1}, "s::2": {"d0": 1}})
    assert summary["failed_task_ids"] == ["s::2"]
    assert summary["methods"]["current/qwen3_rerank"]["top5"]["mean_recall"] == .5
    assert summary["methods"]["current/hybrid_rerank"]["top5"]["mean_recall"] == 1.
    assert summary["methods"]["current/hybrid_rerank"]["elapsed_ms"]["mean"] == 5
    assert rows == original and not summary["answer_quality_evaluated"]


def make_files(tmp_path):
    rows, manifest, contexts = source_data()
    source, corpus = tmp_path/"source", tmp_path/"corpus"
    source.mkdir()
    corpus.mkdir()
    (source/"retrieval.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (source/"manifest.json").write_text(json.dumps(manifest))
    (corpus/"passages.jsonl").write_text("".join(json.dumps(asdict(c))+"\n" for c in contexts))
    (corpus/"manifest.json").write_text(json.dumps({"corpus_sha256": "fixture", "files": {
        "passages.jsonl": file_source(corpus/"passages.jsonl")["sha256"]}}))
    return source, corpus


def test_dry_run_does_not_load_models_indexes_or_qrels(tmp_path, monkeypatch):
    source, corpus = make_files(tmp_path)
    monkeypatch.setattr(rerank_replay, "LocalQwen3Reranker", lambda *a, **k: pytest.fail("model must not load"))
    monkeypatch.setattr(rerank_replay, "load_qrels", lambda *a: pytest.fail("labels must not load"))
    from adapters.hybrid_retrieval import DenseIndex
    from adapters.retrieval import BM25SRetriever
    monkeypatch.setattr(DenseIndex, "load", lambda *a, **k: pytest.fail("no dense retrieval"))
    monkeypatch.setattr(BM25SRetriever, "load", lambda *a, **k: pytest.fail("no BM25 retrieval"))
    output = tmp_path/"dry"
    rerank_replay.main(["--source-run", str(source), "--corpus-index", str(corpus), "--pool-k", "5",
                       "--output", str(output), "--dry-run", "--qrels", "absent.tsv"])
    assert json.loads((output/"manifest.json").read_text())["status"] == "dry_run"
    assert len((output/"pools.jsonl").read_text().splitlines()) == 2
    assert not (output/"reranked.jsonl").exists()


def test_cli_score_then_qrels_records_failure_and_never_overwrites(tmp_path, monkeypatch):
    source, corpus = make_files(tmp_path)
    scorer = ScorerFixture(fail=True)
    monkeypatch.setattr(rerank_replay, "LocalQwen3Reranker", lambda *a, **k: scorer)
    qrels = tmp_path/"qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\ns::1\td0\t1\ns::2\td0\t1\n")
    loader = rerank_replay.load_qrels
    def after_scoring(path):
        assert len(scorer.calls) == 2
        return loader(path)
    monkeypatch.setattr(rerank_replay, "load_qrels", after_scoring)
    output = tmp_path/"scored"
    args = ["--source-run", str(source), "--corpus-index", str(corpus), "--pool-k", "5",
            "--output", str(output), "--model-path", "prepared-local", "--qrels", str(qrels)]
    rerank_replay.main(args)
    assert scorer.closed
    manifest = json.loads((output/"manifest.json").read_text())
    assert manifest["status"] == "completed_with_failures"
    assert manifest["completed_stage_rows"] == 2
    assert manifest["sources"]["retrieval"] == file_source(source/"retrieval.jsonl")
    assert json.loads((output/"summary.json").read_text())["failed_task_ids"] == ["s::2"]
    with pytest.raises(FileExistsError):
        rerank_replay.main(args)


def test_changed_passage_body_and_unequal_token_budget_are_rejected(tmp_path):
    source, corpus = make_files(tmp_path)
    with pytest.raises(SystemExit):
        rerank_replay.main(["--source-run", str(source), "--corpus-index", str(corpus),
            "--output", str(tmp_path/"unequal"), "--pool-k", "5", "--max-length", "512", "--dry-run"])
    with (corpus/"passages.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="passage file differs"):
        rerank_replay.main(["--source-run", str(source), "--corpus-index", str(corpus),
            "--output", str(tmp_path/"changed"), "--pool-k", "5", "--dry-run"])
