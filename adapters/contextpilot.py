"""Read-only ContextPilot principle adapter, without its mutable server pipeline.

The overlap/position metric and matched-prefix alignment follow ContextPilot
(https://arxiv.org/abs/2511.03475). This is a small,
independent adaptation over committed evidence IDs; it does not reproduce their
hierarchical clustering, batch scheduling or cross-session cache index.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from controller.schema import EvidenceBlock, EvidenceKey, SessionSnapshot


def context_distance(left: tuple[EvidenceKey, ...], right: tuple[EvidenceKey, ...],
                     alpha: float = .001) -> float:
    """ContextPilot's overlap fraction plus mean shared-item position difference."""
    left_positions = {key: index for index, key in enumerate(left)}
    shared = [(left_positions[key], index) for index, key in enumerate(right)
              if key in left_positions]
    if not shared:
        return 1.0
    return (1 - len(shared) / max(len(left), len(right)) +
            alpha * sum(abs(a - b) for a, b in shared) / len(shared))


def committed_contexts(snapshot: SessionSnapshot) -> tuple[tuple[EvidenceKey, ...], ...]:
    """Read committed source declaration order, then older history when available."""
    contexts = []
    if snapshot.execution:
        by_message: dict[int, list[EvidenceKey]] = defaultdict(list)
        for span in snapshot.execution.evidence_spans:
            index = span.message_index if span.alias_declaration is None else span.alias_declaration[0]
            by_message[index].append(span.key)
        contexts.extend(tuple(keys) for _, keys in sorted(by_message.items()))
    actual_sets = {frozenset(context) for context in contexts}
    contexts.extend(turn.evidence_keys for turn in snapshot.raw_history
                    if turn.evidence_keys and frozenset(turn.evidence_keys) not in actual_sets)
    return tuple(dict.fromkeys(contexts))


def order_evidence(blocks: Iterable[EvidenceBlock], snapshot: SessionSnapshot,
                   retrieval_order: Iterable[EvidenceKey] = ()) -> tuple[EvidenceBlock, ...]:
    """Align with the closest committed context, then retain retrieval order.

    Dependencies missing from retrieval order use stable-key order. This index is
    local to the candidate call; a failed generation cannot teach it new context.
    """
    by_key = {block.key: block for block in blocks}
    initial = tuple(dict.fromkeys(key for key in retrieval_order if key in by_key))
    initial += tuple(sorted(by_key.keys() - set(initial)))
    contexts = committed_contexts(snapshot)
    if not contexts:
        return tuple(by_key[key] for key in initial)
    matched = min(contexts, key=lambda context: context_distance(initial, context))
    shared = tuple(key for key in matched if key in by_key)
    ordered = tuple(dict.fromkeys((*shared, *initial)))
    return tuple(by_key[key] for key in ordered)


def reuse_annotation(reference: str) -> str:
    """An annotation is valid only if the caller established current visibility."""
    return f"Refer to evidence [{reference}] already supplied in this conversation."


def retrieval_order_annotation(references: Iterable[str]) -> str:
    """Keep per-query relevance order separate from the cache-aligned body order."""
    return "Retrieval priority (most relevant first): " + " > ".join(
        f"[{reference}]" for reference in references) + "."
