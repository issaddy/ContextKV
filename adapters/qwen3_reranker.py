"""Local Qwen3 reranking adapted from the fixed, Apache-2.0 official model card.

Source: Qwen/Qwen3-Reranker-0.6B README at fd9fb1d26c07223ced488065909faf522e29cc7d.
The official prompt, formatting and final-token yes/no scoring are retained.
Changes: local-only loading, explicit device/dtype, exact truncation accounting,
batched scoring, and Transformers logits_to_keep=1/use_cache=False for inference.
See docs/third_party/QWEN3_RERANKER_NOTICE.md for attribution and license.
"""
from __future__ import annotations

from pathlib import Path


MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "fd9fb1d26c07223ced488065909faf522e29cc7d"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
          'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
          '<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_instruction(query, document, instruction=DEFAULT_INSTRUCTION):
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def prepare_inputs(tokenizer, texts, prefix_tokens, suffix_tokens, max_length):
    """Preserve the complete official wrapper; cap the single-string body on the right."""
    body_budget = max_length-len(prefix_tokens)-len(suffix_tokens)
    if body_budget < 1:
        raise ValueError("max_length must leave room for the official prompt body")
    # One untruncated tokenization yields exact length diagnostics. For each
    # single input string, right slicing is the official longest_first truncation.
    encoded = tokenizer(texts, padding=False, truncation=False, return_attention_mask=False)
    lengths, inputs = [], []
    for tokens in encoded["input_ids"]:
        used = prefix_tokens+tokens[:body_budget]+suffix_tokens
        inputs.append(used)
        lengths.append({"original_input_tokens": len(tokens)+len(prefix_tokens)+len(suffix_tokens),
                        "used_input_tokens": len(used), "truncated": len(tokens) > body_budget})
    return tokenizer.pad({"input_ids": inputs}, padding=True, return_tensors="pt"), lengths


def yes_no_probabilities(logits, no_id, yes_id, torch):
    # Left padding makes the last position the final real prompt token for every row.
    last = logits[:, -1, :]
    binary = torch.stack([last[:, no_id], last[:, yes_id]], dim=1).float()
    return torch.nn.functional.log_softmax(binary, dim=1)[:, 1].exp().tolist()


class LocalQwen3Reranker:
    def __init__(self, model_path, *, device="cpu", dtype="float32", batch_size=8,
                 max_length=1024, instruction=DEFAULT_INSTRUCTION):
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch size and token limit must be positive")
        import torch
        from importlib.metadata import version
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.batch_size, self.max_length = torch, batch_size, max_length
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                                       padding_side="left", truncation_side="right")
        self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        if max_length <= len(self.prefix_tokens)+len(self.suffix_tokens):
            raise ValueError("max_length must leave room for the official prompt body")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.no_id == self.yes_id or self.no_id is None or self.yes_id is None:
            raise ValueError("the local tokenizer must expose distinct no/yes token IDs")
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                                                         dtype=getattr(torch, dtype))
        self.model.to(device=device).eval()
        self.texts_processed, self.truncated_pairs, self.last_pairs = 0, 0, []
        self.manifest = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
            "model_path": str(Path(model_path).resolve()), "revision_provenance": "operator_supplied_local_snapshot",
            "license": "Apache-2.0", "device": str(self.model.device), "dtype": str(self.model.dtype),
            "max_length": max_length, "batch_size": batch_size, "instruction": instruction,
            "prefix": PREFIX, "suffix": SUFFIX, "padding_side": "left", "body_truncation_side": "right",
            "prefix_tokens": len(self.prefix_tokens), "suffix_tokens": len(self.suffix_tokens),
            "no_token_id": self.no_id, "yes_token_id": self.yes_id,
            "scoring": "exp(log_softmax([last_no_logit,last_yes_logit]))[yes]; not calibrated relevance",
            "logits_to_keep": 1, "use_cache": False,
            "packages": {name: version(name) for name in ("torch", "transformers")}}

    def score(self, query, contexts):
        scores, self.last_pairs = [], []
        for start in range(0, len(contexts), self.batch_size):
            batch = contexts[start:start+self.batch_size]
            texts = [format_instruction(query, c.title+"\n"+c.text, self.instruction) for c in batch]
            inputs, lengths = prepare_inputs(self.tokenizer, texts, self.prefix_tokens, self.suffix_tokens, self.max_length)
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                outputs = self.model(**inputs, logits_to_keep=1, use_cache=False, return_dict=True)
                scores.extend(yes_no_probabilities(outputs.logits, self.no_id, self.yes_id, self.torch))
            self.last_pairs.extend({"document_id": c.document_id, **length} for c, length in zip(batch, lengths))
            self.texts_processed += len(batch)
            self.truncated_pairs += sum(length["truncated"] for length in lengths)
        return scores

    def close(self):
        device = self.model.device
        del self.model
        if device.type == "cuda":
            with self.torch.cuda.device(device):
                self.torch.cuda.empty_cache()
