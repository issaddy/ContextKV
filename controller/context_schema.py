"""Shared task, source, support and reading contracts.

Coordinates are half-open Unicode character offsets into original text. Budgets
use tokenizer IDs and milliseconds. Runtime handles in RetrievalState are not
persisted; plans and step records contain only JSON data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    def __post_init__(self):
        for name in ("token_ids", "provenance", "dependencies", "anchors"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.span is not None:
            object.__setattr__(self, "span", tuple(self.span))


@dataclass(frozen=True)
class TextRef:
    turn_id: int
    speaker: str
    start: int
    end: int


@dataclass(frozen=True)
class SourceSpan:
    span_id: str
    source_key: EvidenceKey
    start: int
    end: int


@dataclass(frozen=True)
class Condition:
    condition_id: str
    key_span: TextRef | None = None
    value_spans: tuple[TextRef, ...] = ()
    relation_type: str = "PREDICATE"
    predicate_span: TextRef | None = None
    target_refs: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    update: str = "ADD"


@dataclass(frozen=True)
class Need:
    need_id: str
    anchor_spans: tuple[TextRef, ...] = ()
    operation: str = "lookup"
    subject_refs: tuple[TextRef, ...] = ()
    condition_ids: tuple[str, ...] = ()
    predecessor_need_ids: tuple[str, ...] = ()
    scope: str = "selected_sources"


@dataclass(frozen=True)
class VersionIntent:
    mode: str = "current"
    document_families: tuple[str, ...] = ()
    version_ids: tuple[str, ...] = ()
    time_refs: tuple[TextRef, ...] = ()
    source_turns: tuple[int, ...] = ()
    temporal_relation: str = "at"
    resolved_version_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskState:
    task_id: str
    raw_question: str
    working_history: tuple[TextRef, ...] = ()
    conditions: tuple[Condition, ...] = ()
    needs: tuple[Need, ...] = ()
    version_intent: VersionIntent = field(default_factory=VersionIntent)
    source_scope: tuple[EvidenceKey, ...] = ()
    current_turn_id: int = 0
    query: str = ""
    effective_source_scope: tuple[EvidenceKey, ...] | None = None


@dataclass(frozen=True)
class RetrievalState:
    snapshot_version: int
    committed_history: tuple[Any, ...] = ()
    prior_task_states: tuple[TaskState, ...] = ()
    archive_source_keys: tuple[EvidenceKey, ...] = ()
    source_index: Any = None
    committed_readings: tuple[tuple[str, ReadingOutcome], ...] = ()
    # None observes the public directory; an explicit tuple bounds supplied sources.
    source_scope: tuple[EvidenceKey, ...] | None = None


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    source_spans: tuple[SourceSpan, ...]
    text: str
    heading: str = ""
    table_id: str | None = None
    dependencies: tuple[str, ...] = ()
    applicability: tuple[tuple[str, str], ...] = ()
    body_tokens: int = 0
    retrieval_score: float = 0.0
    document_family: str = ""
    valid_from: str | None = None
    valid_to: str | None = None
    published_at: str | None = None
    need_ids: tuple[str, ...] = ()
    original_passage_ids: tuple[str, ...] = ()
    version_order: tuple[int, ...] = ()
    current: bool | None = None


@dataclass(frozen=True)
class EvidencePool:
    units: tuple[EvidenceUnit, ...]
    source_blocks: tuple[EvidenceBlock, ...] = ()
    scope_id: str = "selected_sources"
    retrieval_ms: float = 0.0
    trace: tuple[dict[str, Any], ...] = ()
    auxiliary_ms: tuple[tuple[str, float], ...] = ()
    auxiliary_usage: dict[str, Any] | None = None
    selected_set: EvidenceSet | None = None


@dataclass(frozen=True)
class SupportBundle:
    bundle_id: str
    need_id: str
    unit_ids: tuple[str, ...]
    applicability: float
    joint_support: float
    source_spans: tuple[SourceSpan, ...] = ()
    disposition: str = "partial"
    bridge_value: float = 0.0
    missing_relation: str = ""


@dataclass(frozen=True)
class SupportScores:
    singles: tuple[SupportBundle, ...] = ()
    bundles: tuple[SupportBundle, ...] = ()
    scoring_ms: float = 0.0


@dataclass(frozen=True)
class EvidenceSet:
    set_id: str
    selected_units: tuple[EvidenceUnit, ...]
    dependency_closure: tuple[str, ...] = ()
    need_support: tuple[SupportBundle, ...] = ()
    quality_prediction: float = 0.0
    token_cost: int = 0
    disposition: str = "answer"
    search_ms: float = 0.0
    source_blocks: tuple[EvidenceBlock, ...] = ()


@dataclass(frozen=True)
class OpenNeed:
    need_id: str
    missing_relation: str
    bound_entities: tuple[str, ...] = ()
    temporal_scope: str = "current"
    parent_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str
    need_id: str
    kind: str
    text: str = ""
    value: Any = None
    unit: str = ""
    conditions: tuple[str, ...] = ()
    source_spans: tuple[SourceSpan, ...] = ()
    operation: str = "extract"
    parent_claim_ids: tuple[str, ...] = ()
    entity: str = ""
    relation: str = ""
    version: str = ""
    shared_need_ids: tuple[str, ...] = ()

    def serves(self, need_id):
        return need_id == self.need_id or need_id in self.shared_need_ids


@dataclass(frozen=True)
class NumericReadScope:
    scope_id: str
    need_ids: tuple[str, ...]
    condition_ids: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReadTask:
    step_id: str
    kind: str
    need_ids: tuple[str, ...] = ()
    input_unit_ids: tuple[str, ...] = ()
    input_claim_ids: tuple[str, ...] = ()
    predecessor_step_ids: tuple[str, ...] = ()
    input_budget: int = 12288
    output_budget: int = 2048
    output_schema: dict[str, Any] = field(default_factory=dict)
    operation: str = "extract"
    reduction_basis_claim_ids: tuple[str, ...] = ()
    shared_numeric_scopes: tuple[NumericReadScope, ...] = ()
    retained_conclusion_tokens: tuple[tuple[str, int], ...] = ()
    requires_reduction: bool = False
    condition_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReductionState:
    """Completed logical reduction, independently of its physical model calls."""
    step_id: str
    need_ids: tuple[str, ...]
    operation: str
    basis_claim_ids: tuple[str, ...]
    frontier_claim_ids: tuple[str, ...]
    mode: str


@dataclass(frozen=True)
class ReadResult:
    step_id: str
    claims: tuple[EvidenceClaim, ...] = ()
    covered_units: tuple[str, ...] = ()
    support: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    open_needs: tuple[OpenNeed, ...] = ()
    raw_output: str = ""
    elapsed_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reduction: ReductionState | None = None


@dataclass(frozen=True)
class AnswerPoint:
    need_id: str
    claim_ids: tuple[str, ...] = ()
    source_spans: tuple[SourceSpan, ...] = ()
    citation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerPlan:
    points: tuple[AnswerPoint, ...] = ()
    disposition: str = "answer"


@dataclass(frozen=True)
class ContextBudget:
    window_tokens: int = 32768
    body_tokens: int = 8192
    history_tokens: int = 8192
    output_tokens: int = 1024
    read_input_tokens: int = 12288
    read_output_tokens: int = 2048
    max_steps: int = 64
    max_total_tokens: int | None = 1048576
    max_passages: int | None = None
    max_complete_ms: float | None = None
    lambda_time: float = 0.0
    lambda_gpu: float = 0.0
    lambda_host: float = 0.0


@dataclass(frozen=True)
class AnswerWindowAllocation:
    fixed_input_tokens: int
    conclusion_tokens: int
    read_output_tokens: int
    need_tokens: tuple[tuple[str, int], ...] = ()
    source_metadata_tokens: int = 0


@dataclass(frozen=True)
class ContextPlan:
    plan_id: str
    task_state: TaskState
    selected_set: EvidenceSet
    coverage_scope: str = "selected_sources"
    steps: tuple[ReadTask, ...] = ()
    answer_plan: AnswerPlan = field(default_factory=AnswerPlan)
    budget: ContextBudget = field(default_factory=ContextBudget)
    predicted_cost_ms: float = 0.0
    coverage_units: tuple[str, ...] = ()
    auxiliary_ms: tuple[tuple[str, float], ...] = ()
    candidate_qualities: tuple[tuple[str, str, float], ...] = ()
    layouts: tuple[str, ...] = ("existing", "need_order", "stable_evidence")
    cost_prediction_eligible: bool = True
    reading_mode: str = "typed_reduce"
    online_tokens_spent: int | None = 0
    separate_needs: bool = False
    numeric_identity: str = "names"
    read_layout: str = "task_first"
    read_packing: str = "mixed"
    answer_system_prompt: str | None = None
    answer_allocation: AnswerWindowAllocation | None = None


@dataclass(frozen=True)
class Message:
    role: str
    content: str


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
    actual_output_tokens: int | None = None


@dataclass(frozen=True)
class StepRecord:
    attempt_id: str
    step_id: str
    task: ReadTask
    engine_epoch: ContextEpoch | None = None
    input_messages: tuple[Message, ...] = ()
    input_ids: tuple[int, ...] = ()
    output_ids: tuple[int, ...] = ()
    result: ReadResult | None = None
    receipt: CompletionReceipt | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReadingOutcome:
    plan: ContextPlan
    results: tuple[ReadResult, ...] = ()
    steps: tuple[StepRecord, ...] = ()
    claims: tuple[EvidenceClaim, ...] = ()
    coverage: tuple[tuple[str, str, str, str], ...] = ()
    open_needs: tuple[OpenNeed, ...] = ()
    auxiliary_usage: dict[str, Any] | None = None
    answer_representation: str = "existing"
