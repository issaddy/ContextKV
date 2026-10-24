"""Shared P3 retriever construction; default remains the existing pinned BM25S."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import ExitStack
from pathlib import Path
import time

import yaml

from adapters.mtrag import DATASET_COMMIT
from adapters.retrieval import BM25SRetriever, load_beir_corpus


@dataclass(frozen=True)
class HybridRetrievalConfig:
    sparse_index: str
    dense_index: str
    encoder_path: str
    schema_version: int = 1
    kind: str = "hybrid"
    stage: str = "hybrid_rerank"
    query_mode: str = "current"
    query_fusion: str = "single"
    reranker_path: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 16
    encoder_max_length: int = 8192
    reranker_max_length: int = 1024
    candidate_k: int = 50
    rerank_k: int = 50
    rrf_k: int = 60
    cpu_threads: int = 1
    selector: dict | None = None
    selector_history_turns: int = 3
    selector_schema_mode: str = "static_v1"
    rerank_query_mode: str = "resolved"


def load_retriever_config(path):
    """JSON is a YAML subset. Relative asset paths follow other CLI paths: cwd."""
    value = HybridRetrievalConfig(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if value.schema_version != 1 or value.kind != "hybrid":
        raise ValueError("retriever config requires schema_version=1 and kind=hybrid")
    from adapters.query_resolution import SCHEMA_MODES, SOURCE_MODES
    if value.query_mode not in {"current", "last3", "all_user", "correction_aware", "user_topic", *SOURCE_MODES}:
        raise ValueError("online query_mode must use user questions: current/last3/all_user/correction_aware/user_topic")
    if value.stage not in {"dense", "hybrid", "hybrid_rerank"}:
        raise ValueError("unknown retrieval stage")
    if value.query_fusion not in {"single", "current_topic_rrf"}:
        raise ValueError("unknown query fusion")
    if value.query_fusion == "current_topic_rrf" and (value.query_mode not in {"user_topic", *SOURCE_MODES} or value.stage == "dense"):
        raise ValueError("current_topic_rrf requires user_topic/source_topic/assistant_topic_unverified and hybrid/hybrid_rerank")
    if value.query_mode in SOURCE_MODES:
        from adapters.query_selector import SelectorConfig
        if value.query_fusion != "current_topic_rrf" or value.selector is None:
            raise ValueError("topic selector modes require selector config and current_topic_rrf")
        SelectorConfig(**value.selector)
        if value.selector_history_turns < 1:
            raise ValueError("selector_history_turns must be positive")
        if value.selector_schema_mode not in SCHEMA_MODES:
            raise ValueError("unknown selector schema mode")
    elif value.selector is not None:
        raise ValueError("selector config is only used by explicit topic selector modes")
    elif value.selector_schema_mode != "static_v1":
        raise ValueError("selector_schema_mode is only used by explicit topic selector modes")
    if value.stage == "hybrid_rerank" and not value.reranker_path:
        raise ValueError("hybrid_rerank requires reranker_path")
    if value.rerank_query_mode not in {"resolved", "current", "user_topic"}:
        raise ValueError("unknown rerank query mode")
    if value.stage != "hybrid_rerank" and value.rerank_query_mode != "resolved":
        raise ValueError("rerank_query_mode requires hybrid_rerank")
    if value.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("unsupported model dtype")
    if min(value.batch_size, value.encoder_max_length, value.reranker_max_length, value.cpu_threads) < 1:
        raise ValueError("batch size, token limits and CPU threads must be positive")
    if value.candidate_k < 5 or value.rerank_k < 5 or value.rrf_k < 0:
        raise ValueError("retrieval budgets must cover five passages and a nonnegative RRF constant")
    return value


def retriever_config_sources(path):
    """Small provenance files for the existing experiment-boundary digest mechanism."""
    config = load_retriever_config(path)
    return (str(path), str(Path(config.sparse_index)/"manifest.json"),
            str(Path(config.dense_index)/"manifest.json"))


def create_retriever(protocol, *, config_path=None, index_path=None, corpus_path=None, trace_sink=None):
    if protocol != "P3":
        return None
    if config_path:
        if index_path or corpus_path:
            raise ValueError("choose --retriever-config or legacy --retriever-index/--corpus")
        from adapters.hybrid_retrieval import DenseIndex, HybridRetriever, LocalBGE
        started = time.perf_counter()
        config = load_retriever_config(config_path)
        sparse = BM25SRetriever.load(config.sparse_index, cpu_threads=config.cpu_threads)
        dense = DenseIndex.load(config.dense_index)
        # Ownership transfers to HybridRetriever only after complete construction.
        # Failed startup closes the models/client already created on this path.
        with ExitStack() as cleanup:
            encoder = LocalBGE(config.encoder_path, device=config.device, dtype=config.dtype,
                               batch_size=config.batch_size, max_length=config.encoder_max_length)
            cleanup.callback(encoder.close)
            reranker = None
            if config.stage == "hybrid_rerank":
                reranker = LocalBGE(config.reranker_path, reranker=True, device=config.device, dtype=config.dtype,
                                   batch_size=config.batch_size, max_length=config.reranker_max_length)
                cleanup.callback(reranker.close)
            resolver = None
            if config.selector is not None:
                from adapters.query_resolution import TopicResolver
                from adapters.query_selector import OpenAICompletionClient, SelectorConfig
                client = OpenAICompletionClient(SelectorConfig(**config.selector))
                cleanup.callback(client.close)
                resolver = TopicResolver(client, config.query_mode, history_turns=config.selector_history_turns,
                                         schema_mode=config.selector_schema_mode)
            result = HybridRetriever(sparse, dense, encoder, reranker, query_mode=config.query_mode,
                stage=config.stage, candidate_k=config.candidate_k, rerank_k=config.rerank_k, rrf_k=config.rrf_k,
                query_fusion=config.query_fusion,
                rerank_query_mode=config.rerank_query_mode,
                resolver=resolver, trace_sink=trace_sink,
                source_config={"path": str(Path(config_path).resolve()), "resolved": asdict(config),
                               "relative_path_base": str(Path.cwd())})
            cleanup.pop_all()
        result.initialization_seconds = time.perf_counter()-started
        return result
    if index_path:
        return BM25SRetriever.load(index_path)
    if corpus_path:
        return BM25SRetriever(load_beir_corpus(corpus_path), corpus_snapshot_id=f"mtrag-{DATASET_COMMIT}")
    raise ValueError("P3 requires --corpus, --retriever-index or --retriever-config")
