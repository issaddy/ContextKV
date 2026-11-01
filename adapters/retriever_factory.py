"""Shared P3 retriever construction; default remains the existing pinned BM25S."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import ExitStack
from pathlib import Path
import time

import yaml

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
    rerank_query_mode: str = "resolved"
    source_window: bool = False
    query_adapter_path: str | None = None
    source_documents: str | None = None
    evidence_selection: str = "ranked"


def load_retriever_config(path):
    """JSON is a YAML subset. Relative asset paths follow other CLI paths: cwd."""
    value = HybridRetrievalConfig(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if value.schema_version != 1 or value.kind != "hybrid":
        raise ValueError("retriever config requires schema_version=1 and kind=hybrid")
    from adapters.hybrid_retrieval import validate_retrieval_options
    validate_retrieval_options(query_mode=value.query_mode, stage=value.stage,
        query_fusion=value.query_fusion, rerank_query_mode=value.rerank_query_mode,
        candidate_k=value.candidate_k, rerank_k=value.rerank_k, rrf_k=value.rrf_k,
        joint_selection=value.evidence_selection == "semantic")
    if value.evidence_selection not in {"ranked", "semantic"}:
        raise ValueError("evidence_selection must be ranked or semantic")
    if value.evidence_selection == "semantic" and (
            value.query_mode != "conversational" or value.selector is None):
        raise ValueError("semantic evidence selection requires conversational mode with selector")
    if value.query_adapter_path and value.query_mode not in {"current", "dialogue"}:
        raise ValueError("query adapters require their trained literal current/dialogue view")
    if value.query_mode in {"conversational", "dialogue"}:
        from adapters.query_selector import SelectorConfig
        if value.selector is not None:
            SelectorConfig(**value.selector)
    elif value.selector is not None:
        raise ValueError("selector config requires conversational or dialogue mode")
    if value.stage == "hybrid_rerank" and not value.reranker_path:
        raise ValueError("hybrid_rerank requires reranker_path")
    if value.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("unsupported model dtype")
    if min(value.batch_size, value.encoder_max_length, value.reranker_max_length, value.cpu_threads) < 1:
        raise ValueError("batch size, token limits and CPU threads must be positive")
    return value


def create_retriever(protocol, *, config_path=None, index_path=None, corpus_path=None, trace_sink=None,
                     corpus_snapshot_id=None):
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
                               batch_size=config.batch_size, max_length=config.encoder_max_length,
                               query_adapter_path=config.query_adapter_path,
                               normalize_float32=config.query_mode == "dialogue")
            cleanup.callback(encoder.close)
            if config.query_adapter_path and encoder.manifest["query_adapter"]["query_mode"] != config.query_mode:
                raise ValueError("query adapter input recipe differs from configured query_mode")
            reranker = None
            if config.stage == "hybrid_rerank":
                reranker = LocalBGE(config.reranker_path, reranker=True, device=config.device, dtype=config.dtype,
                                   batch_size=config.batch_size, max_length=config.reranker_max_length)
                cleanup.callback(reranker.close)
            resolver, client = None, None
            if config.selector is not None:
                from adapters.query_selector import OpenAICompletionClient, SelectorConfig
                client = OpenAICompletionClient(SelectorConfig(**config.selector))
                cleanup.callback(client.close)
            if config.query_mode == "conversational" or client is not None:
                from adapters.conversation import ConversationResolver
                resolver = ConversationResolver(client)
            evidence_selector = None
            if config.evidence_selection == "semantic":
                from adapters.evidence_selection import EvidenceSelector
                evidence_selector = EvidenceSelector(resolver)
            documents = None
            if config.source_window and config.source_documents:
                from adapters.source_documents import SourceDocuments
                documents = SourceDocuments.load(config.source_documents)
            result = HybridRetriever(sparse, dense, encoder, reranker, query_mode=config.query_mode,
                stage=config.stage, candidate_k=config.candidate_k, rerank_k=config.rerank_k, rrf_k=config.rrf_k,
                query_fusion=config.query_fusion,
                rerank_query_mode=config.rerank_query_mode,
                source_window=config.source_window,
                source_documents=documents,
                evidence_selector=evidence_selector,
                resolver=resolver, trace_sink=trace_sink,
                source_config={"path": str(Path(config_path).resolve()), "resolved": asdict(config),
                               "relative_path_base": str(Path.cwd())})
            cleanup.pop_all()
        result.initialization_seconds = time.perf_counter()-started
        return result
    if index_path:
        return BM25SRetriever.load(index_path)
    if corpus_path:
        return BM25SRetriever(load_beir_corpus(corpus_path),
                             corpus_snapshot_id=corpus_snapshot_id or Path(corpus_path).stem)
    raise ValueError("P3 requires --corpus, --retriever-index or --retriever-config")
