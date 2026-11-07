"""ContextPilot ordering: committed-history baseline and optional author CPU index.

The baseline follows the paper's overlap/position alignment. SourceOrderIndex
uses pinned author source through a narrow index-only loader; no author prompt
pipeline, inference hooks, or GPU computation is enabled by this adapter.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Iterable

from controller.schema import CompletionReceipt, EvidenceBlock, EvidenceKey, SessionSnapshot


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


# Optional source-only integration. See docs/third_party/CONTEXTPILOT_NOTICE.md.
CONTEXTPILOT_COMMIT = "b4c80327f6cfae2286839413731fcb6f53578b2f"


def load_contextpilot_index(checkout):
    """Load the audited CPU index from fixed source, without package setup/hooks.

    Deliberately bypass upstream's package initializer, which imports its RAG
    pipeline and unrelated model/retriever dependencies. No source is rewritten.
    """
    import importlib
    import sys
    from types import ModuleType

    root = _verified_checkout(str(checkout))
    for name, directory in (("contextpilot", root / "contextpilot"),
                            ("contextpilot.server", root / "contextpilot/server")):
        previous = sys.modules.get(name)
        if previous is not None:
            if tuple(getattr(previous, "__path__", ())) != (str(directory),):
                raise ValueError("another ContextPilot package is already imported")
        else:
            package = ModuleType(name)
            package.__path__ = [str(directory)]
            package.__package__ = name
            sys.modules[name] = package
    return importlib.import_module("contextpilot.server.live_index").ContextPilot


class SourceOrderIndex:
    """Executed-source ordering hints only; native quotes determine actual reuse.

    Source IDs must identify compiler-visible versioned original windows, including
    coordinates and content identity. This adapter never changes a prompt, reads
    source bodies, or treats a context-index match as physical cache residency.
    """
    def __init__(self, checkout):
        from threading import RLock
        self._index_type = load_contextpilot_index(checkout)
        self._scopes = {}
        self._executed = {}
        self._lock = RLock()

    @staticmethod
    def cache_scope(fingerprint, cache_config_hash, engine_restart_id):
        from controller.serialization import dumps
        from controller.schema import ConfigFingerprint
        if not isinstance(fingerprint, ConfigFingerprint):
            raise ValueError("source index requires the actual model/template fingerprint")
        if not cache_config_hash or not engine_restart_id:
            raise ValueError("source index requires a real cache config and engine restart")
        return dumps((fingerprint, cache_config_hash, engine_restart_id))

    @staticmethod
    def _encode(source_ids):
        import hashlib
        source_ids = tuple(source_ids)
        if any(not isinstance(identity, str) or not identity for identity in source_ids):
            raise ValueError("source IDs must be nonempty stable strings")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source windows must have unique IDs within a context")
        values = tuple(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
                       & ((1 << 63)-1) for identity in source_ids)
        return source_ids, values

    def propose_orders(self, source_ids, *, fingerprint, cache_config_hash, engine_restart_id):
        """Read-only alternatives: original first, then at most one index alignment.

        Upstream search/traverse update diagnostic counters, but do not insert
        requests or update LRU access times. Unselected candidates teach nothing.
        """
        scope = self.cache_scope(fingerprint, cache_config_hash, engine_restart_id)
        original, encoded = self._encode(source_ids)
        with self._lock:
            index = self._scopes.get(scope)
            if index is None or not encoded:
                return (original,)
            path, node_id, overlap, _ = index.search(list(encoded), update_access=False)
            if node_id < 0 or not overlap:
                return (original,)
            node = index.traverse(path)
            prefix = index.metadata[node_id].doc_ids
            if node is None or not prefix:
                return (original,)
            reordered = tuple(index._reorder_with_prefix(list(encoded), prefix))
            decode = dict(zip(encoded, original, strict=True))
            proposed = tuple(decode[value] for value in reordered)
            return (original,) if proposed == original else (original, proposed)

    def observe_executed(self, source_ids, *, fingerprint, cache_config_hash,
                         receipt: CompletionReceipt, input_ids):
        """Register only a completed, hash-bound actual execution, idempotently.

        The caller supplies the actual compiler descriptor's ordered source IDs;
        its mapping to text/token ranges remains the compiler's responsibility.
        """
        from controller.templates import token_hash
        if not isinstance(receipt, CompletionReceipt):
            raise TypeError("source index requires this project's CompletionReceipt")
        scope = self.cache_scope(fingerprint, cache_config_hash, receipt.engine_restart_id)
        original, encoded = self._encode(source_ids)
        if (not receipt.generation_complete or not input_ids
                or token_hash(input_ids) != receipt.input_hash or not receipt.attempt_id):
            raise ValueError("source index requires a completed actual input-bound receipt")
        if not original:
            return None
        key = (scope, receipt.attempt_id)
        binding = (receipt.input_hash, original)
        with self._lock:
            previous = self._executed.get(key)
            if previous is not None:
                if previous[:2] != binding:
                    raise ValueError("executed source binding changed for an existing attempt")
                return previous[2]
            index = self._scopes.get(scope)
            if index is None:
                index = self._index_type(use_gpu=False)
                built = index.build_and_schedule([list(encoded)], len(input_ids))
                if built["reordered_contexts"] != [list(encoded)]:
                    raise ValueError("initial index changed the executed source order")
                request_id = built["request_ids"][0]
                self._scopes[scope] = index
            else:
                path, _, _, _ = index.search(list(encoded), update_access=False)
                _, _, request_id = index.insert(list(encoded), path, len(input_ids))
            self._executed[key] = (*binding, request_id)
            return request_id


@lru_cache(maxsize=None)
def _verified_checkout(checkout):
    """Verify the prepared source revision once per process, never install it."""
    from pathlib import Path
    import subprocess
    root = Path(checkout).resolve()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != CONTEXTPILOT_COMMIT:
        raise ValueError("ContextPilot checkout is not the fixed author commit")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError("ContextPilot checkout has modified tracked source")
    return root
