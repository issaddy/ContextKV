"""Build BM25 and frozen Qwen document vectors for the shared retrieval pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


class QwenDocumentEncoder:
    def __init__(self, model_path, *, device="cpu", dtype="float32", max_length=32768):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.model = AutoModel.from_pretrained(model_path, dtype=getattr(torch, dtype)).to(device).eval()
        self.manifest = {"model_id": "Qwen/Qwen3-Embedding-0.6B", "revision": "base", "max_length": max_length,
                         "pooling": "last_nonpadding_token_l2", "model_path": str(model_path)}

    def encode(self, texts):
        return self.encode_with_usage(texts)[0]

    def encode_with_usage(self, texts):
        import torch
        batch = self.tokenizer(texts, padding=True, return_tensors="pt")
        input_tokens = int(batch["attention_mask"].sum())
        batch = batch.to(self.model.device)
        with torch.no_grad():
            states = self.model(**batch).last_hidden_state
            return torch.nn.functional.normalize(states[:, -1].float(), dim=-1).cpu().numpy(), input_tokens

    def close(self):
        self.model = None


def main():
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
