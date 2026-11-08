"""Read-only causal-prefix observations for joint plan selection.

The quote is the plan's cache snapshot. A later owner-status observation adds
ownership provenance; neither observation reserves KV or predicts its cost.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from bisect import bisect_left, bisect_right
import hashlib
import json

from controller.templates import longest_common_prefix, token_hash
from engine_patches.contextkv_runtime import executed_prefix_hashes


def native_cache_scope_hash(fingerprint, cache_config_hash):
    return hashlib.sha256(json.dumps([asdict(fingerprint), cache_config_hash],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PrefixMatchView:
    prefix_id: str
    kind: str
    matched_tokens: int
    full_owner_prefix: bool
    causal_page_ids: tuple[str, ...]
    quote_gpu_page_ids: tuple[str, ...]
    owner_gpu_page_ids: tuple[str, ...]
    owner_host_page_ids: tuple[str, ...]
    reference_present: bool | None
    evidence_attribution: str = "unmeasured"


@dataclass(frozen=True)
class CandidatePrefixView:
    candidate_id: str
    input_hash: str
    cache_scope_hash: str
    quote_seq: int
    quote_time_ns: int
    engine_restart_id: str
    page_size: int
    matches: tuple[PrefixMatchView, ...]
    quote_ready_gpu_tokens: int
    quote_recoverable_tokens: int
    quote_host_load_tokens: int
    transfer_pending: bool
    resource_cost_status: str = "unmeasured"

    @property
    def prefix_ids(self):
        return tuple(match.prefix_id for match in self.matches)


def prefix_candidate_view(candidates, quote_batch, prefix_status, owner_records=(), *, page_size):
    """Map an exact native quote and native owners to actual candidate paths.

    Shared causal pages have the same ID across owners and business sessions.
    The independent owner contributes its observed replica-page sets only;
    quote readiness remains bound to the quote sequence. System/catalogue pages
    are deliberately not relabelled as evidence without source-body offsets.
    """
    if page_size <= 0:
        raise ValueError("native page_size must be positive")
    if quote_batch.fingerprint is None:
        raise ValueError("prefix view needs the actual quoted reader fingerprint")
    if prefix_status["engine_restart_id"] != quote_batch.engine_restart_id:
        raise ValueError("owner status and quote belong to different native restarts")
    fingerprint = asdict(quote_batch.fingerprint)
    scope = native_cache_scope_hash(quote_batch.fingerprint, quote_batch.cache_config_hash)
    owners = {record.owner_id: record for record in owner_records}
    observed = []
    for status in prefix_status["prefixes"]:
        if (status["engine_restart_id"] != quote_batch.engine_restart_id
                or status["cache_config_hash"] != quote_batch.cache_config_hash
                or status["fingerprint"] != fingerprint or status["cache_scope_hash"] != scope
                or status["page_size"] != page_size):
            raise ValueError("native owner status differs from the quoted reader/cache scope")
        record = owners.get(status["owner_id"])
        if record is None:
            continue  # Other clients' token paths are not inferred from a path hash.
        ids = record.prefix_input_ids
        if (record.receipt.engine_restart_id != quote_batch.engine_restart_id
                or asdict(record.fingerprint) != fingerprint
                or len(ids) != status["boundary_tokens"] or len(ids) % page_size
                or token_hash(ids) != status["prefix_hash"]):
            raise ValueError("durable owner token path differs from its actual native status")
        expected = {scope+":"+value for value in executed_prefix_hashes(ids, page_size).values()}
        if not set(status["gpu_page_ids"]) | set(status["host_page_ids"]) <= expected:
            raise ValueError("native replica pages differ from the owner's causal token path")
        observed.append((record, status))
    quotes = {quote.candidate_id: quote for quote in quote_batch.quotes}
    result = {}
    for candidate in candidates:
        quote = quotes[candidate.candidate_id]
        if (candidate.fingerprint != quote_batch.fingerprint
                or token_hash(candidate.input_ids) != candidate.input_hash
                or quote.input_hash != candidate.input_hash
                or quote.input_len != len(candidate.input_ids)):
            raise ValueError("candidate input differs from its exact native quote")
        limit = min(quote.match_limit, quote.recoverable_prefix_tokens,
                    len(candidate.input_ids))//page_size*page_size
        hashes = executed_prefix_hashes(candidate.input_ids[:limit], page_size)
        page_ids = {end: scope+":"+hashed for end, hashed in hashes.items()}
        gpu_limit = min(limit, quote.ready_gpu_prefix_tokens)//page_size*page_size
        matches = []
        if limit:
            matches.append(PrefixMatchView("native:"+scope+":"+hashes[limit],
                "native_quote", limit, False, tuple(page_ids.values()),
                tuple(page for end, page in page_ids.items() if end <= gpu_limit), (), (), None))
        for record, status in observed:
            matched = min(limit, longest_common_prefix(candidate.input_ids, record.prefix_input_ids),
                status["residency"]["recoverable_prefix_tokens"])//page_size*page_size
            if not matched:
                continue
            actual = tuple(page for end, page in page_ids.items() if end <= matched)
            allowed = set(actual)
            matches.append(PrefixMatchView(record.owner_id, "independent_owner", matched,
                matched == len(record.prefix_input_ids), actual,
                tuple(page for end, page in page_ids.items() if end <= min(matched, gpu_limit)),
                tuple(page for page in status["gpu_page_ids"] if page in allowed),
                tuple(page for page in status["host_page_ids"] if page in allowed),
                status["reference_present"]))
        result[candidate.candidate_id] = CandidatePrefixView(candidate.candidate_id,
            candidate.input_hash, scope, quote_batch.observed_seq, quote_batch.observed_time_ns,
            quote_batch.engine_restart_id, page_size, tuple(matches), quote.ready_gpu_prefix_tokens,
            quote.recoverable_prefix_tokens, quote.host_only_load_tokens,
            bool(quote.inflight_h2d_tokens or quote.inflight_d2h_tokens))
    return result


def source_fragment_id(source_spans, text, encoding):
    """Identity of original coordinates and actual body encoding, never task IDs."""
    from .serialization import dumps
    coordinates = tuple((span.source_key, span.start, span.end) for span in source_spans)
    return "source:"+hashlib.sha256(dumps((encoding, coordinates, text)).encode()).hexdigest()


def rendered_message_starts(rendered, messages):
    """Locate actual message contents in the exact rendered request."""
    starts, cursor = [], 0
    for message in messages:
        start = rendered.text.index(message.content, cursor)
        starts.append(start)
        cursor = start+len(message.content)
    return tuple(starts)


def rendered_token_ranges(template, rendered, char_ranges):
    from .templates import ByteTokenizer
    if rendered.offsets is not None:
        offsets = rendered.offsets
    elif type(template.tokenizer) is ByteTokenizer:
        offsets = ((index, index+1) for index, char in enumerate(rendered.text) for _ in char.encode("utf-8"))
    else:
        encoded = template.tokenizer(rendered.text, add_special_tokens=False, return_offsets_mapping=True)
        if tuple(encoded["input_ids"]) != rendered.input_ids:
            raise ValueError("source geometry tokenizer differs from the compiled request")
        offsets = encoded["offset_mapping"]
    starts, ends, positions = [], [], []
    for position, (first, last) in enumerate(offsets):
        if last > first:
            starts.append(first)
            ends.append(last)
            positions.append(position)
    ranges = []
    for first, last in char_ranges:
        if last <= first:
            continue
        left, right = bisect_right(ends, first), bisect_left(starts, last)
        if left < right:
            ranges.append((positions[left], positions[right-1]+1))
    return ranges


def actual_source_geometry(template, candidate, rendered=None):
    """Measure the source union and its causal boundary on the actual token path."""
    from .source_prompt import SOURCE_SYSTEM

    ranges, source_prefix_tokens = candidate.source_body_token_ranges, candidate.source_prefix_tokens
    if ranges is None and candidate.evidence_spans:
        rendered = template.render(candidate.messages, with_offsets=True) if rendered is None else rendered
        if rendered.input_ids != candidate.input_ids or rendered.input_hash != candidate.input_hash:
            raise ValueError("source geometry differs from the compiled request")
        starts = rendered_message_starts(rendered, candidate.messages)
        intervals = [
            (starts[span.message_index]+span.start, starts[span.message_index]+span.end)
            for span in candidate.evidence_spans]
        source_first = (candidate.messages[0] == SOURCE_SYSTEM
                        and candidate.evidence_spans[0].message_index == 1)
        if source_first:
            intervals.append((starts[1], starts[1]+len(candidate.messages[1].content)))
        ranges = rendered_token_ranges(template, rendered, intervals)
        if source_first:
            source_prefix_tokens = ranges.pop()[1]
    merged = []
    for first, last in sorted(set(ranges or ())):
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(last, merged[-1][1]))
        else:
            merged.append((first, last))
    boundary = max((last for _, last in merged), default=0)
    return {"body_token_ranges": tuple(merged), "body_tokens": sum(last-first for first, last in merged),
            "causal_boundary_tokens": boundary, "causal_prefix_input_ids": candidate.input_ids[:boundary],
            "causal_prefix_hash": token_hash(candidate.input_ids[:boundary]),
            "source_prefix_tokens": source_prefix_tokens}
