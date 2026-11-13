"""Configured providers for ordinary RAG, LlamaIndex and shared ContextKV."""
from __future__ import annotations

import time

from adapters.corpus import CorpusIndex
from adapters.evidence_pool import SourceDirectory
from adapters.hybrid_retrieval import DenseIndex, reciprocal_rank_fusion
from adapters.jina_encoding import pack_passages, project_markers, merge_rankings
from adapters.retrieval import BM25SRetriever, run_retrieval_worker
from controller.schema import EvidenceBlock, OnlineContext
from controller.cooperative_request import EvidenceSelection


class ConventionalEvidenceProvider:
    """One hybrid recall and released Jina rerank for the classic QA references."""
    def __init__(self, config, template):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from adapters.shared_encoding import QwenDocumentEncoder
        retrieval, models = config["retrieval"], config["shared"]
        self.config, self.template = config, template
        query = models["query"]
        self.encoder = QwenDocumentEncoder(query["model_path"], device=query["device"],
                                           dtype=query.get("dtype", "bfloat16"))
        evidence = models["evidence"]
        self.reranker = AutoModel.from_pretrained(evidence["model_path"], trust_remote_code=True,
            dtype=getattr(torch, evidence.get("dtype", "bfloat16"))).to(evidence["device"]).eval()
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(evidence["model_path"], trust_remote_code=True)
        self.rerank_tokens = evidence.get("batch_tokens", 32768)
        if retrieval.get("corpus_path"):
            self.sparse = BM25SRetriever.load(retrieval["sparse_index"])
            self.dense = DenseIndex.load(retrieval["dense_index"])
            self.corpus_snapshot_id = self.sparse.corpus_snapshot_id
            self.corpus = CorpusIndex.from_jsonl(retrieval["corpus_path"], corpus_id=retrieval["corpus_id"],
                                                snapshot_id=self.corpus_snapshot_id)
        else:
            self.sparse = self.dense = None
            self.corpus_snapshot_id = "supplied"
            self.corpus = CorpusIndex((), corpus_id=retrieval["corpus_id"], snapshot_id=self.corpus_snapshot_id)
        self.candidate_k = retrieval.get("candidate_k", 50)
        self.top_k = config.get("selection", {}).get("max_passages") or 5
        self._supplied = {}

    @property
    def manifest(self):
        return {"retrieval": "bm25_qwen_embedding_rrf_jina_rerank", "top_k": self.top_k,
                "query": self.encoder.manifest, "evidence": "jinaai/jina-reranker-v3.5",
                "corpus_snapshot_id": self.corpus_snapshot_id}

    async def retrieve(self, snapshot, query, request_id):
        return await run_retrieval_worker(self._retrieve, snapshot, query, request_id)

    async def retrieve_supplied(self, snapshot, query, request_id, blocks, *, source_metadata=None,
                                coverage_scope="selected_sources"):
        return await run_retrieval_worker(self._retrieve, snapshot, query, request_id,
                                          blocks=tuple(blocks), metadata=source_metadata)

    def _retrieve(self, snapshot, query, request_id, *, blocks=None, metadata=None):
        started = time.perf_counter()
        input_tokens = 0
        sparse, dense, corpus, directory = self.sparse, self.dense, self.corpus, None
        if blocks is not None:
            key = tuple(block.key for block in blocks)
            if key not in self._supplied:
                import numpy as np
                corpus = CorpusIndex(blocks, corpus_id=self.corpus.corpus_id,
                    snapshot_id=self.corpus_snapshot_id, source_metadata=metadata)
                directory = SourceDirectory(corpus, self.template.tokenizer)
                contexts = tuple(OnlineContext(unit.unit_id, unit.text, title=unit.heading)
                                 for unit in directory.units_by_id.values())
                sparse = BM25SRetriever({item.document_id: {"text": item.text, "title": item.title}
                    for item in contexts}, corpus_snapshot_id=self.corpus_snapshot_id)
                arrays = [self.encoder.encode_with_usage([item.title+"\n"+item.text for item in contexts[index:index+16]])
                          for index in range(0, len(contexts), 16)]
                input_tokens += sum(tokens for _, tokens in arrays)
                dense = DenseIndex(contexts, np.concatenate([values for values, _ in arrays]), {})
                self._supplied[key] = sparse, dense, corpus, directory
            sparse, dense, corpus, directory = self._supplied[key]
        vector, tokens = self.encoder.encode_with_usage(["Instruct: Retrieve evidence for the question.\nQuery: "+query])
        input_tokens += tokens
        dense_rows = dense.search_vectors(vector, top_k=self.candidate_k)[0]
        sparse_rows = sparse.retrieve([query], top_k=self.candidate_k)
        fused = reciprocal_rank_fusion([[item.document_id for item in group] for group in (dense_rows, sparse_rows)])
        candidates = tuple(dense.by_id[identity] for identity, _ in fused[:self.candidate_k])
        selected, tokens = self._rerank(query, candidates)
        input_tokens += tokens
        if directory is None:
            evidence = corpus.resolve(corpus.retrieved_keys(selected))
        else:
            evidence = []
            for item in selected:
                unit = directory.units_by_id[item.document_id]
                for span in unit.source_spans:
                    original = corpus.resolve((span.source_key,))[0]
                    base = original.span[0] if original.span else 0
                    evidence.append(EvidenceBlock(original.key, original.source_uri, original.title,
                        original.text[span.start-base:span.end-base], span=(span.start, span.end),
                        parent_section=original.parent_section, provenance=original.provenance))
            evidence = tuple(evidence)
        usage = {"compute_ms": (time.perf_counter()-started)*1000, "input_tokens": input_tokens, "output_tokens": 0}
        return EvidenceSelection(evidence, auxiliary_usage=usage)

    def _rerank(self, query, contexts):
        import inspect
        import torch
        format_prompt = inspect.getmodule(self.reranker).format_docs_prompts_func
        texts = [context.title+"\n"+context.text for context in contexts]
        groups = pack_passages(texts,
            [len(self.rerank_tokenizer.encode(text))+20 for text in texts],
            overhead=2*len(self.rerank_tokenizer.encode(query))+200, budget=self.rerank_tokens)
        rankings, tokens = [], 0
        with torch.no_grad():
            for texts in groups:
                prompt = format_prompt(query, list(texts), special_tokens=self.reranker.special_tokens)
                batch = self.rerank_tokenizer(prompt, return_tensors="pt").to(self.reranker.device)
                tokens += batch["input_ids"].numel()
                states = self.reranker.model(**batch, use_cache=False).last_hidden_state[0]
                ids = batch["input_ids"][0]
                rankings.append(project_markers(self.reranker, states,
                    ids == self.reranker.doc_embed_token_id,
                    (ids == self.reranker.query_embed_token_id).nonzero()[0, 0]))
            if not rankings:
                return (), 0
            _, scores, _ = merge_rankings(rankings)
            order = scores.argsort(descending=True)[:self.top_k].tolist()
        return tuple(contexts[index] for index in order), tokens

    def close(self):
        self.encoder.close()
        self.reranker = None


def create_evidence_provider(config, template):
    if config.get("execution", {}).get("dialogue_mode", "contextkv") == "contextkv":
        from adapters.shared_retrieval import SharedEvidenceProvider
        return SharedEvidenceProvider(config, template)
    return ConventionalEvidenceProvider(config, template)
