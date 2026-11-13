"""Build BM25 and frozen Qwen document vectors for the shared retrieval pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    from adapters.shared_encoding import QwenDocumentEncoder
    from adapters.hybrid_retrieval import DenseIndex
    from adapters.retrieval import BM25SRetriever, load_beir_corpus
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    output = Path(args.output)
    sparse = BM25SRetriever(load_beir_corpus(args.corpus), corpus_snapshot_id=args.snapshot)
    sparse.save(output / "sparse")
    encoder = QwenDocumentEncoder(args.model, device=args.device, dtype=args.dtype)
    DenseIndex.build(sparse._contexts, encoder, output / "dense", corpus_snapshot_id=args.snapshot,
                     batch_size=args.batch_size)
    print(json.dumps({"passages": len(sparse._contexts), "sparse": str(output / "sparse"), "dense": str(output / "dense")}))


if __name__ == "__main__":
    main()
