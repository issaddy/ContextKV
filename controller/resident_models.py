"""Stable identities of actually loaded models, indexes and source catalogs."""
from __future__ import annotations

import hashlib
import json

from .serialization import to_jsonable


# Request policy and live counters do not identify a resident allocation.
_RUNTIME = frozenset({"budget", "budgets", "strategy", "selection", "reading", "planner",
    "lambda_time", "lambda_gpu", "lambda_host", "http_calls", "calls", "completed_retrievals",
    "initialization_seconds", "query_view_traces", "model_usage", "usage", "last_metrics",
    "elapsed_ms", "complete_ms", "compute_ms", "input_tokens", "output_tokens", "token_usage",
    "max_total_tokens", "max_total_model_tokens", "max_complete_ms", "max_passages", "max_steps",
    "window_tokens", "body_tokens", "history_tokens", "output_budget", "input_budget",
    "read_input_tokens", "read_output_tokens", "max_input_tokens", "max_new_tokens", "context_window",
    "max_length", "batch_size", "temperature", "top_p", "top_k", "min_p", "timeout_seconds",
    "wire_schema", "input_policy", "token_limit_hits", "texts_processed"})


def _asset(value, *, strip_runtime=True, model_scope=False):
    if isinstance(value, dict):
        return {key: ("cuda" if key == "device" and isinstance(item, str)
                      and item.startswith("cuda:") and item[5:].isdecimal()
                      else _asset(item, strip_runtime=strip_runtime, model_scope=key == "model_scope"))
                for key, item in sorted(value.items())
                if key not in {"engine_restart_id", "base_url"} and not (model_scope and key == "source_path")
                and (not strip_runtime or key not in _RUNTIME)}
    if isinstance(value, (tuple, list)):
        return [_asset(item, strip_runtime=strip_runtime) for item in value]
    return value


def resident_model_identity(manifest):
    """Extract residency from a raw StudyEvidenceProvider manifest.

    Keep the original manifest in the capture/audit artifact. The extracted
    value binds corpus/index identities, actual model manifests/deployments and
    observed helper engine identities, independently of request budgets, process
    restarts, service routes and the path used to load a model-scope record.
    """
    raw = to_jsonable(manifest)
    if raw is None:
        return None
    if raw.get("format") == "contextkv_resident_assets_v1":
        return _asset(raw, strip_runtime=False)
    if "retrieval" not in raw:
        return {"format": "contextkv_resident_assets_v1", "assets": _asset(raw)}
    retrieval = raw["retrieval"]
    retriever = retrieval.get("retriever") or {}
    assets = {key: _asset(retriever[key]) for key in (
        "kind", "corpus_snapshot_id", "sparse", "dense", "encoder", "reranker",
        "task_interpretation", "semantic_evidence_selection") if key in retriever}
    source = retriever.get("source_window")
    if source is not None:
        assets["source_document_catalog"] = {key: source[key] for key in (
            "mode", "document_alignment_counts") if key in source}
    return {"format": "contextkv_resident_assets_v1",
        "retrieval": {**{key: retrieval[key] for key in ("corpus_id", "corpus_snapshot_id", "metadata_enabled")
                          if key in retrieval}, "assets": assets},
        **{key: _asset(raw[key]) for key in ("task_model", "support_model", "history_encoder",
            "resident_runtime_models", "actual_helper_info", "helper_engine_info") if key in raw}}


def resident_model_digest(manifest):
    identity = resident_model_identity(manifest)
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def evidence_selection_contract(manifest):
    """Identify A's actual selection policy independently of the resident corpus."""
    if manifest is None:
        raise ValueError("observed gains require the actual evidence selection manifest")
    mode = manifest.get("selection_mode", "support")
    if mode not in {"ranked", "support"}:
        raise ValueError("unknown evidence selection mode in the actual manifest")
    contract = {"mode": mode, "order": manifest.get("evidence_selection_order")}
    if mode == "support":
        contract.update(utility=manifest.get("selection_utility"),
            rerank_retained_sets=manifest.get("rerank_retained_sets", False),
            support_model=_asset(to_jsonable(manifest.get("support_model"))))
    return contract
