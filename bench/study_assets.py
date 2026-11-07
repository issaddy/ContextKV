"""Materialize public evaluation assets independently from training exports."""
from __future__ import annotations

from pathlib import Path


def prepare_public_assets(config):
    from adapters.hybrid_retrieval import DenseIndex, LocalBGE
    from adapters.retrieval import BM25SRetriever, load_beir_corpus
    from adapters.retriever_factory import load_retriever_config
    from bench.versionqa import SOURCE_COMMIT, prepare
    from controller.config import create_model_template, load_config
    from controller.schema import OnlineContext

    settings = config["assets"]["versionqa"]
    target = Path(config["study"]["output_dir"])/"evaluation"/"versionqa"
    retrieval = load_retriever_config(settings["retriever_config"])
    reader = load_config(config["generation"]["run_config"])
    tokenizer = create_model_template(reader.model).tokenizer
    manifest = prepare(settings["source"], settings["questions"], target, tokenizer,
                       pdf_text=settings.get("pdf_text"))
    corpus = load_beir_corpus(target/"corpus.jsonl")
    snapshot = f"versionqa-{SOURCE_COMMIT}-{manifest['corpus_sha256'][:16]}"
    sparse_path, dense_path = target/"bm25s", target/"bge-m3"
    sparse = BM25SRetriever(corpus, corpus_snapshot_id=snapshot, cpu_threads=retrieval.cpu_threads)
    sparse.save(sparse_path)
    encoder = LocalBGE(retrieval.encoder_path, device=retrieval.device, dtype=retrieval.dtype,
        batch_size=retrieval.batch_size, max_length=retrieval.encoder_max_length,
        sdpa_backend=retrieval.sdpa_backend)
    try:
        DenseIndex.build([OnlineContext(**row) for row in corpus.values()], encoder,
                         dense_path, corpus_snapshot_id=snapshot, show_progress=True)
    finally:
        encoder.close()
    return {**manifest, "corpus_snapshot_id": snapshot, "sparse_index": str(sparse_path),
            "dense_index": str(dense_path)}
