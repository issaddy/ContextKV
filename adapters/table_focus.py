"""Optional lexical row focus, with literal source IDs on every fact line.

CRAFT's MIT row-text builder is reused; its dense reranking is not reproduced.
Requires the existing retrieval extra only when this optional style is selected.
Selection reads only the current user query and source table fields. It never
receives task IDs, answers, relevance labels, or generated assistant history.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import re

from adapters.query_views import correction_cue
from adapters.table_relations import TableRelations
from adapters._craft_row_text import build_row_texts


def _terms(text):
    try:
        import bm25s
    except ImportError as exc:
        raise RuntimeError("focus_refs requires contextkv[retrieval] (BM25S)") from exc
    vocabulary = bm25s.tokenize([text], stopwords="en", show_progress=False).vocab
    def singular(word):
        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            return word[:-1]
        return word
    return {singular(word) for word in vocabulary}


def query_scope(query):
    """Reuse the existing correction detector, narrowing only a delimited lead-in."""
    cue = correction_cue(query)
    scoped, reason = query, "current_query"
    if cue and cue.lower() in {"no", "not"} and not re.match(r"\s*not\s+only\b", query, re.I):
        pieces = re.split(r"[,;:]", query, maxsplit=1)
        if len(pieces) == 2 and pieces[1].strip():
            scoped, reason = pieces[1].strip(), "after_explicit_correction_leadin"
    broad = re.search(r"\b(?:all|both|every|total|combined|sum|average|difference|compare|comparison)\b", scoped, re.I)
    unresolved_negative = re.search(r"\b(?:not|no|without|except|excluding)\b|n't\b", scoped, re.I)
    preserve_all = "multirow_request" if broad else "unresolved_negation" if unresolved_negative else None
    return {"original_query": query, "query_used_for_focus": scoped, "correction_cue": cue,
            "scope_reason": reason, "preserve_all_reason": preserve_all}


def select_rows(query: str, table: TableRelations):
    scope = query_scope(query)
    terms = _terms(scope["query_used_for_focus"])
    # Adaptation boundary: CRAFT supplies this row text + row metadata. The
    # following lexical scoring replaces its dense encoder, for this lightweight style.
    has_paths = any(r.parent_path for r in table.relations)
    row_texts, row_meta = build_row_texts([{
        "table_id": f"table_at_{table.span.start}",
        "title": table.caption.text if table.caption else "",
        "headers": ["row", "column", "value"] + (["explicit source parent path"] if has_paths else []),
        "rows": [[r.object.text, r.column.text, r.value.text]
                 + ([" > ".join(p.text for p in r.parent_path)] if has_paths else []) for r in table.relations],
    }])
    scores = [(len(terms & _terms(r.object.text)), len(terms & _terms(r.column.text)),
               len(terms & _terms(row_text))) for r, row_text in zip(table.relations, row_texts)]
    best = max(scores, default=(0, 0, 0))
    preserve = scope["preserve_all_reason"] or ("no_matching_terms" if best == (0, 0, 0) else None)
    selected = [i for i, score in enumerate(scores) if preserve or score == best]
    return tuple(table.relations[i] for i in selected), {**scope, "query_terms": sorted(terms),
        "scoring": "lexicographic distinct-term overlap: object, column, CRAFT row text; preserve all maximum ties",
        "all_rows_reason": preserve, "selected_indices": selected,
        "rows": [{"index": i, "score": list(score), "craft_row_meta": row_meta[i], "source": asdict(relation)}
                 for i, (relation, score) in enumerate(zip(table.relations, scores))]}


def focused_annotation(query: str, tables_by_reference: tuple[tuple[str, tuple[TableRelations, ...]], ...]):
    """Return a small current-turn supplement plus auditable selection traces."""
    lines, traces = [], []
    for reference, tables in tables_by_reference:
        for table in tables:
            selected, trace = select_rows(query, table)
            traces.append({"reference_id": reference, "table_span": [table.span.start, table.span.end], **trace})
            for row in selected:
                column_label = ("original table header" if table.syntax in {"captioned_numeric_rows", "unnumbered_numeric_rows"}
                                else "parent list label" if table.syntax == "nested_numeric_list" else "column")
                lines.append(f"- [{reference}] object={json.dumps(row.object.text, ensure_ascii=False)}; "
                    f"{column_label}={json.dumps(row.column.text, ensure_ascii=False)}; "
                    f"value={json.dumps(row.value.text, ensure_ascii=False)}.")
                if row.parent_path:
                    lines.append(f"  [{reference}] explicit source parent path: "
                                 + " > ".join(json.dumps(p.text, ensure_ascii=False) for p in row.parent_path))
                for note in row.notes:
                    lines.append(f"  [{reference}] source note with matching row marker: {json.dumps(note.text, ensure_ascii=False)}")
            linked = {note for row in selected for note in row.notes}
            for note in table.source_notes:
                if note not in linked:
                    lines.append(f"- [{reference}] adjacent source note; row scope unspecified: {json.dumps(note.text, ensure_ascii=False)}")
    if not lines:
        return "", traces
    return ("Source table facts relevant to the current question (the complete source remains above). "
            "Each fact's original Evidence ID is shown inline; use it when citing that fact. "
            "The row associations add no cross-row inclusion rule or performance guarantee.\n"
            + "\n".join(lines)), traces

