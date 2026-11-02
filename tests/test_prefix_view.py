"""Joint planner sees actual quoted paths, not a retention value estimate."""
from dataclasses import asdict, replace
import hashlib
import json

import pytest

from controller.prefix_retention import PrefixOwnerRecord
from controller.prefix_view import prefix_candidate_view
from controller.schema import CacheQuote, CacheQuoteBatch, CompletionReceipt, ConfigFingerprint, PromptCandidate
from controller.templates import token_hash
from engine_patches.contextkv_runtime import executed_prefix_hashes


def fixture():
    fp = ConfigFingerprint()
    ids = tuple(range(1, 13))
    receipt = CompletionReceipt("rid", "source", "epoch", 1, "restart", token_hash(ids), "stop", True)
    record = PrefixOwnerRecord("owner", "session", "turn", "read:0", receipt, fp,
        ids, (), "value", {}, 30, stage="RETAINED")
    scope = hashlib.sha256(json.dumps([asdict(fp), "recipe"], sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    pages = tuple(scope+":"+item for item in executed_prefix_hashes(ids, 2).values())
    owner = dict(owner_id="owner", engine_restart_id="restart", cache_config_hash="recipe",
        fingerprint=asdict(fp), cache_scope_hash=scope, prefix_hash=token_hash(ids),
        boundary_tokens=12, page_size=2, gpu_page_ids=list(pages[:3]), host_page_ids=list(pages[2:]),
        reference_present=True, residency={"recoverable_prefix_tokens": 12})
    candidate_ids = ids[:8]+(99, 100, 101)
    candidate = PromptCandidate("read", "rebuild", 0, (), candidate_ids, token_hash(candidate_ids),
        frozenset(), frozenset(), fingerprint=fp, generation_kind="READ")
    quote = CacheQuote("read", candidate.input_hash, len(candidate_ids), 10,
        raw_match_tokens=8, ready_gpu_prefix_tokens=6, recoverable_prefix_tokens=8,
        host_only_load_tokens=2)
    batch = CacheQuoteBatch("q", 7, 123, "restart", "recipe", 100, 100, (quote,), fp)
    return candidate, batch, {"engine_restart_id": "restart", "prefixes": [owner]}, record, pages


def test_partial_owner_match_uses_actual_pages_and_preserves_quote_scope():
    candidate, batch, status, record, pages = fixture()
    view = prefix_candidate_view((candidate,), batch, status, (record,), page_size=2)["read"]
    natural, retained = view.matches
    assert natural.kind == "native_quote" and retained.prefix_id == "owner"
    assert retained.matched_tokens == 8 and not retained.full_owner_prefix
    assert retained.causal_page_ids == pages[:4]
    assert retained.quote_gpu_page_ids == retained.owner_gpu_page_ids == pages[:3]
    assert retained.owner_host_page_ids == pages[2:4]
    assert view.quote_seq == 7 and view.resource_cost_status == "unmeasured"
    assert retained.evidence_attribution == "unmeasured"


def test_shared_causal_pages_have_same_ids_and_no_unmatched_owner_is_invented():
    candidate, batch, status, record, pages = fixture()
    second = replace(record, owner_id="same-pages-other-session", session_id="other")
    status["prefixes"].append({**status["prefixes"][0], "owner_id": second.owner_id})
    view = prefix_candidate_view((candidate,), batch, status, (record, second), page_size=2)["read"]
    assert view.matches[1].causal_page_ids == view.matches[2].causal_page_ids == pages[:4]
    assert len(set(page for match in view.matches for page in match.causal_page_ids)) == 4
    bare = prefix_candidate_view((candidate,), batch, status, (), page_size=2)["read"]
    assert len(bare.matches) == 1 and bare.matches[0].kind == "native_quote"


def test_current_quote_and_native_status_require_real_restart_scope_and_input():
    candidate, batch, status, record, _ = fixture()
    with pytest.raises(ValueError, match="restarts"):
        prefix_candidate_view((candidate,), batch, {**status, "engine_restart_id": "new"},
            (record,), page_size=2)
    with pytest.raises(ValueError, match="exact native quote"):
        prefix_candidate_view((replace(candidate, input_ids=(99,)),), batch, status, (record,), page_size=2)
    status["prefixes"][0]["gpu_page_ids"].append("wrong-causal-page")
    with pytest.raises(ValueError, match="causal token path"):
        prefix_candidate_view((candidate,), batch, status, (record,), page_size=2)
