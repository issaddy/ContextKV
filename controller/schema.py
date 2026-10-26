"""Immutable contracts shared by the controller, adapters and engine bridge.

All token counts refer to actual tokenizer IDs.  Timing values are milliseconds;
cache capacities are token slots, never reserved process memory.
"""
from __future__ import annotations

from dataclasses import dataclass


def _tuples(instance, *names: str) -> None:
    for name in names:
        object.__setattr__(instance, name, tuple(getattr(instance, name)))


def _references(instance) -> None:
    object.__setattr__(instance, "reference_map",
                       tuple((str(ref), key) for ref, key in instance.reference_map))


@dataclass(frozen=True, order=True)
class EvidenceKey:
    corpus_id: str
    document_id: str
    passage_id: str
    version: str
    content_sha256: str


@dataclass(frozen=True)
class EvidenceBlock:
    key: EvidenceKey
    source_uri: str
    title: str
    text: str
    token_ids: tuple[int, ...] = ()
    span: tuple[int, int] | None = None
    parent_section: str | None = None
    provenance: tuple[str, ...] = ()
    dependencies: tuple[EvidenceKey, ...] = ()
    anchors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _tuples(self, "token_ids", "provenance", "dependencies", "anchors")
        if self.span is not None:
            object.__setattr__(self, "span", tuple(self.span))


@dataclass(frozen=True)
class EvidenceSpan:
    """Complete body location and provenance; aliases also locate their declaration."""
    key: EvidenceKey
    message_index: int
    start: int
    end: int
    source_uri: str
    provenance: tuple[str, ...]
    reference_id: str
    body_reference_id: str | None = None
    alias_declaration: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        _tuples(self, "provenance")
        if self.alias_declaration is not None:
            _tuples(self, "alias_declaration")


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class TaskFrame:
    """Open-valued conversational state; describes a request, never answer facts."""
    topic: str = ""
    target: str = ""
    task: str = ""
    conditions: tuple[tuple[str, str], ...] = ()
    requirements: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", tuple(sorted((name, value) for name, value in self.conditions)))
        _tuples(self, "requirements")


@dataclass(frozen=True)
class EvidenceUse:
    """A derived reading guide pointing into an unchanged original source."""
    key: EvidenceKey
    start: int
    end: int
    needs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _tuples(self, "needs")


@dataclass(frozen=True)
class QueryContext:
    """Derived conversational intent, never an evidence source or a raw QA edit."""
    query: str
    relation: str
    history_turns: tuple[int, ...] = ()
    task_frame: TaskFrame | None = None
    evidence_uses: tuple[EvidenceUse, ...] = ()

    def __post_init__(self) -> None:
        _tuples(self, "history_turns", "evidence_uses")


@dataclass(frozen=True)
class TurnRecord:
    turn_id: int
    user_text: str
    assistant_text: str
    assistant_output_ids: tuple[int, ...] = ()
    finish_reason: str = "stop"
    evidence_keys: tuple[EvidenceKey, ...] = ()
    query_context: QueryContext | None = None

    def __post_init__(self) -> None:
        _tuples(self, "assistant_output_ids", "evidence_keys")


@dataclass(frozen=True)
class ConfigFingerprint:
    model_revision: str = "mock-byte-v1"
    tokenizer_hash: str = "byte-v1"
    template_hash: str = "contextkv-chat-v1"
    namespace_hash: str = "default"
    cache_salt: str | None = None
    extra_key: str | None = None

@dataclass(frozen=True)
class ExecutionLedger:
    epoch_id: int
    engine_session_id: str
    history_version: int
    messages: tuple[Message, ...]
    last_input_ids: tuple[int, ...]
    last_output_ids: tuple[int, ...]
    visible_blocks: frozenset[EvidenceKey]
    template_hash: str
    model_revision: str
    corpus_snapshot_id: str
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    reference_map: tuple[tuple[str, EvidenceKey], ...] = ()
    inserted_cache_tokens: int = 0

    def __post_init__(self) -> None:
        _tuples(self, "messages", "last_input_ids", "last_output_ids", "evidence_spans")
        _references(self)
        object.__setattr__(self, "visible_blocks", frozenset(self.visible_blocks))


@dataclass(frozen=True)
class ContextEpoch:
    business_session_id: str
    epoch_no: int
    engine_session_id: str
    session_generation: int
    parent_epoch_no: int | None = None
    state: str = "PREPARED"
    reason: str = "initial"
    engine_restart_id: str = ""


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    version: int = 0
    raw_history: tuple[TurnRecord, ...] = ()
    raw_history_version: int = 0
    execution: ExecutionLedger | None = None
    archive: tuple[EvidenceBlock, ...] = ()
    current_epoch: ContextEpoch | None = None
    corpus_snapshot_id: str = "demo-v1"
    history_sync_pending: bool = False
    force_new_epoch_for_recovery: bool = False
    turns_since_rebuild: int = 0
    must_rebuild: bool = False
    reference_map: tuple[tuple[str, EvidenceKey], ...] = ()

    def __post_init__(self) -> None:
        _tuples(self, "raw_history", "archive")
        _references(self)


@dataclass(frozen=True)
class PromptCandidate:
    candidate_id: str
    action: str
    snapshot_version: int
    messages: tuple[Message, ...]
    input_ids: tuple[int, ...]
    input_hash: str
    visible_blocks: frozenset[EvidenceKey]
    required_blocks: frozenset[EvidenceKey]
    build_ms: float = 0.0
    feasible: bool = True
    reason: str | None = None
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    reference_map: tuple[tuple[str, EvidenceKey], ...] = ()
    theoretical_lcp: int = 0
    corpus_snapshot_id: str = "demo-v1"
    fingerprint: ConfigFingerprint | None = None
    evidence_resolution: tuple[tuple[EvidenceKey, str], ...] = ()
    evidence_body_tokens: int | None = None
    nonrequired_evidence_body_tokens: int | None = None

    def __post_init__(self) -> None:
        _tuples(self, "messages", "input_ids", "evidence_spans", "evidence_resolution")
        _references(self)
        object.__setattr__(self, "visible_blocks", frozenset(self.visible_blocks))
        object.__setattr__(self, "required_blocks", frozenset(self.required_blocks))


@dataclass(frozen=True)
class QuoteRequest:
    request_id: str
    snapshot_version: int
    model_revision: str
    tokenizer_hash: str
    template_hash: str
    namespace_hash: str
    cache_salt: str | None
    extra_key: str | None
    candidates: tuple[PromptCandidate, ...]

    def __post_init__(self) -> None:
        _tuples(self, "candidates")


@dataclass(frozen=True)
class CacheQuote:
    candidate_id: str
    input_hash: str
    input_len: int
    match_limit: int
    raw_match_tokens: int = 0
    ready_gpu_prefix_tokens: int = 0
    recoverable_prefix_tokens: int = 0
    host_only_load_tokens: int = 0
    inflight_h2d_tokens: int = 0
    inflight_d2h_tokens: int = 0
    native_load_estimate: int = 0
    native_estimate_uncertain: bool = False
    uncertainty_reasons: tuple[str, ...] = ()
    # Ready contiguous GPU prefix currently carrying no native FULL hard lock.
    # None means an older engine did not observe it; zero is a measured value.
    evictable_gpu_prefix_tokens: int | None = None

    def __post_init__(self) -> None:
        _tuples(self, "uncertainty_reasons")


@dataclass(frozen=True)
class NativePrefixMatch:
    """Last native match, not reserved/current residency; server monotonic ns."""
    observed_time_ns: int
    input_tokens: int
    device_prefix_tokens: int
    host_hit_tokens: int
    matched_prefix_tokens: int


@dataclass(frozen=True)
class NativeRequestSummary:
    """Observed CPU-side logical lengths, never unique/releasable physical slots."""
    input_tokens: int
    generated_tokens: int
    max_new_tokens: int
    kv_allocated_tokens: int
    kv_committed_tokens: int
    cache_protected_tokens: int
    ignore_eos: bool = False
    native_finished: bool | None = None
    last_native_match: NativePrefixMatch | None = None
    request_id: str | None = None


@dataclass(frozen=True)
class CacheQuoteBatch:
    request_id: str
    observed_seq: int
    observed_time_ns: int
    engine_restart_id: str
    cache_config_hash: str
    free_gpu_tokens: int
    free_host_tokens: int
    quotes: tuple[CacheQuote, ...]
    fingerprint: ConfigFingerprint | None = None
    evictable_gpu_tokens: int = 0
    evictable_host_tokens: int | None = None
    host_quote_supported: bool = True
    active_requests: int | None = None
    queued_requests: int | None = None
    gpu_capacity_tokens: int | None = None
    native_running_requests: tuple[NativeRequestSummary, ...] | None = None
    native_waiting_requests: tuple[NativeRequestSummary, ...] | None = None
    native_new_token_ratio: float | None = None
    native_max_new_tokens_clip: int | None = None
    native_running_budget_tokens: float | None = None
    native_running_budget_requests: tuple[NativeRequestSummary, ...] | None = None

    def __post_init__(self) -> None:
        _tuples(self, "quotes")
        for name in ("native_running_requests", "native_waiting_requests", "native_running_budget_requests"):
            if getattr(self, name) is not None:
                _tuples(self, name)


@dataclass(frozen=True)
class CompletionReceipt:
    rid: str
    attempt_id: str
    engine_session_id: str
    session_generation: int
    engine_restart_id: str
    input_hash: str
    finish_reason: str
    generation_complete: bool
    actual_device_hit_tokens: int = 0
    actual_host_loaded_tokens: int = 0
    actual_prefill_tokens: int = 0
    inserted_cache_tokens: int = 0
    session_ref_registered: bool = False


@dataclass(frozen=True)
class GenerationResult:
    text: str
    output_ids: tuple[int, ...]
    finish_reason: str
    receipt: CompletionReceipt
    generation_complete: bool = True
    elapsed_ms: float | None = None
    ttft_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None
    queue_ms: float | None = None
    restore_prefill_ms: float | None = None
    tpot_ms: float | None = None
    timing_source: str | None = None
    server_ttft_ms: float | None = None
    server_elapsed_ms: float | None = None
    last_token_ms: float | None = None
    forward_to_first_token_ms: float | None = None

    def __post_init__(self) -> None:
        _tuples(self, "output_ids")


@dataclass(frozen=True)
class ClientTiming:
    """Client monotonic seconds; absent observations remain None, never estimates."""
    t_arrive: float
    t_submit: float | None = None
    t_first: float | None = None
    t_last_token: float | None = None
    t_response_done: float | None = None
    t_commit: float | None = None
    t_retire_ack: float | None = None
    token_observation: str = "none"


@dataclass(frozen=True)
class RetireOperation:
    operation_id: str
    engine_session_id: str
    expected_generation: int
    engine_restart_id: str
    reason: str
    committed_ledger_version: int


@dataclass(frozen=True)
class CostEstimate:
    candidate_id: str
    cost: float
    control_ms: float = 0.0
    queue_ms: float = 0.0
    restore_prefill_ms: float = 0.0
    decode_ms: float = 0.0
    error_ms: float = 0.0
    predicted_output_tokens: int = 1
    p90_output_tokens: int = 1
    risk_cost: float = 0.0
    calibrated: bool = False
    capacity_risk: bool = False
    output_length_error_tokens: float | None = None
    output_length_model: str = "shared"
    output_length_in_range: bool | None = None
    selection_eligible: bool | None = None
    waiting_estimate_kind: str | None = None
    waiting_upper_ms: float | None = None
    waiting_reason: str | None = None
    timing_components_supported: bool | None = None


@dataclass(frozen=True)
class Plan:
    candidate: PromptCandidate
    execution_mode: str = "NATIVE"
    predicted: tuple[CostEstimate, ...] = ()
    quote_seq: int | None = None
    reason: str = ""
    switch_margin_ms: float = 0.0
    predicted_saving_ms: float = 0.0

    def __post_init__(self) -> None:
        _tuples(self, "predicted")


SelectedPlan = Plan


@dataclass(frozen=True)
class NoFeasiblePlan:
    reason: str = "shared_context_policy_overflow"
