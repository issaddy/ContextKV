"""Literal table row/column associations; no model, rewriting, or inferred totals.

Adapts TabRAG's row/column/value/unit representation, not its VLM/LLM pipeline.
All offsets are Python character offsets in the supplied, unchanged passage.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re


@dataclass(frozen=True)
class SourceSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class TableRelation:
    object: SourceSpan
    column: SourceSpan
    value: SourceSpan
    unit: SourceSpan | None
    row_span: SourceSpan
    notes: tuple[SourceSpan, ...] = ()
    parent_path: tuple[SourceSpan, ...] = ()


@dataclass(frozen=True)
class TableRelations:
    syntax: str
    span: SourceSpan
    caption: SourceSpan | None
    headers: tuple[SourceSpan, ...]
    relations: tuple[TableRelation, ...]
    source_notes: tuple[SourceSpan, ...] = ()


_CAPTION = re.compile(r"^Table\s+\d+(?:[.:\-]|\s|$)", re.IGNORECASE)
_SEPARATOR = re.compile(r":?-{3,}:?")
_NUMBER = r"(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"
# Recognize literal unit tokens only. No conversion or unit inheritance from prose.
_UNIT = (r"(?:[KMGTPE]?i?B(?:/s)?|[kMGT]?bps|[kMG]?Hz|[mµμnu]?s|[cmk]?m|[mk]?g|%|"
         r"bytes?|seconds?|minutes?|hours?|days?|tokens?|requests?(?:/s)?)")
_MARKER = r"(?:\[\^[^\]\s]+\]|[†‡]|(?<!\*)\*(?!\*))"
_QUANTITY = re.compile(rf"(?P<number>(?:[<>]=?|[≤≥])?\s*[+\-]?{_NUMBER})\s*(?P<unit>{_UNIT})(?!\w)")
_FLAT_ROW = re.compile(rf"(?P<object>.+?)\s+(?P<value>(?:[<>]=?|[≤≥])?\s*[+\-]?{_NUMBER}\s*"
                       rf"(?P<unit>{_UNIT})(?:{_MARKER})?)\s*$")
_NOTE = re.compile(rf"^(?:Notes?\s*:|Remarks?\s*:|注[：:]|备注[：:]|{_MARKER})", re.IGNORECASE)
_MARKERS = re.compile(_MARKER)


def _span(text: str, start: int, end: int) -> SourceSpan:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return SourceSpan(text[start:end], start, end)


def _cells(text: str, line: SourceSpan) -> tuple[SourceSpan, ...]:
    """Split unescaped Markdown pipes; keep literal escaping inside each cell."""
    pipes, slashes = [], 0
    for i, char in enumerate(line.text):
        if char == "|" and slashes % 2 == 0:
            pipes.append(line.start + i)
        slashes = slashes + 1 if char == "\\" else 0
    if not pipes:
        return ()
    starts, ends = [line.start, *(p + 1 for p in pipes)], [*pipes, line.end]
    cells = [_span(text, a, b) for a, b in zip(starts, ends)]
    if not cells[0].text:
        cells.pop(0)
    if cells and not cells[-1].text:
        cells.pop()
    return tuple(cells)


def _unit(text: str, value: SourceSpan, column: SourceSpan) -> SourceSpan | None:
    matches = list(_QUANTITY.finditer(value.text))
    if len(matches) == 1:
        match = matches[0]
        return _span(text, value.start + match.start("unit"), value.start + match.end("unit"))
    # An explicit unit-only header suffix applies to cells in that column.
    match = re.search(rf"[\[(](?P<unit>{_UNIT})[\])]$", column.text)
    if not matches and match:
        return _span(text, column.start + match.start("unit"), column.start + match.end("unit"))
    return None


def _neighbor_notes(text, lines, start, end, excluded):
    """Copy adjacent explicitly labeled notes, never ordinary following prose."""
    notes = []
    i = start - 1
    while i >= 0 and not lines[i].text:
        i -= 1
    last = i
    while i >= 0 and lines[i].text and i not in excluded:
        i -= 1
    if last >= i + 1 and _NOTE.match(lines[i + 1].text):
        notes.append(_span(text, lines[i + 1].start, lines[last].end))
    i = end
    while i < len(lines):
        while i < len(lines) and not lines[i].text:
            i += 1
        if i == len(lines) or i in excluded or not _NOTE.match(lines[i].text):
            break
        first = i
        i += 1
        while i < len(lines) and lines[i].text and i not in excluded and not _NOTE.match(lines[i].text):
            i += 1
        notes.append(_span(text, lines[first].start, lines[i - 1].end))
    return tuple(notes)


def _with_notes(table, notes):
    relations = []
    for relation in table.relations:
        markers = set(_MARKERS.findall(relation.row_span.text))
        linked = tuple(note for note in notes if (match := _MARKERS.match(note.text))
                       and match.group() in markers)
        relations.append(replace(relation, notes=linked))
    return replace(table, relations=tuple(relations), source_notes=notes)


def extract_table_relations(text: str) -> tuple[TableRelations, ...]:
    """Extract explicit single-header Markdown or captioned numeric row tables.

    Flattened text must have a numbered Table caption, one nonnumeric header,
    and at least two consecutive object/quantity rows with explicit known units.
    Its unsplit header is retained: lost column boundaries are never reconstructed.
    Ragged rows, missing row labels, and fenced/indented code are left untouched.
    """
    lines, excluded, offset, fence = [], set(), 0, None
    for raw in text.splitlines(keepends=True):
        line = _span(text, offset, offset + len(raw))
        index = len(lines)
        marker = re.match(r"(`{3,}|~{3,})", line.text)
        if fence:
            excluded.add(index)
            if marker and marker.group()[0] == fence[0] and len(marker.group()) >= len(fence):
                fence = None
        elif marker:
            fence = marker.group()
            excluded.add(index)
        elif raw.startswith(("    ", "\t")):
            excluded.add(index)
        lines.append(line)
        offset += len(raw)
    tables, i = [], 0
    while i < len(lines):
        if i in excluded or not lines[i].text:
            i += 1
            continue
        headers = _cells(text, lines[i])
        delimiter = _cells(text, lines[i + 1]) if i + 1 < len(lines) else ()
        if (len(headers) >= 2 and len(delimiter) == len(headers) and i + 1 not in excluded
                and all(_SEPARATOR.fullmatch(c.text) for c in delimiter)):
            end, rows, valid = i + 2, [], all(h.text for h in headers)
            valid = valid and len({h.text for h in headers}) == len(headers)
            while end < len(lines) and "|" in lines[end].text and end not in excluded:
                cells = _cells(text, lines[end])
                if len(cells) != len(headers) or not cells[0].text:
                    valid = False
                else:
                    rows.extend(TableRelation(cells[0], column, value, _unit(text, value, column), lines[end])
                                for column, value in zip(headers[1:], cells[1:]))
                end += 1
            if valid:
                previous = i - 1
                while previous >= 0 and not lines[previous].text:
                    previous -= 1
                caption = lines[previous] if previous >= 0 and _CAPTION.match(lines[previous].text) else None
                start = previous if caption else i
                table = TableRelations("markdown", _span(text, lines[start].start, lines[end - 1].end),
                                       caption, headers, tuple(rows))
                tables.append(_with_notes(table, _neighbor_notes(text, lines, start, end, excluded)))
            i = end
            continue
        if _CAPTION.match(lines[i].text):
            header = i + 1
            while header < len(lines) and not lines[header].text:
                header += 1
            start = header + 1
            while start < len(lines) and not lines[start].text:
                start += 1
            end, rows, valid = start, [], header < len(lines) and header not in excluded
            if valid:
                valid = "|" not in lines[header].text and not re.search(r"\d", lines[header].text)
            while end < len(lines) and lines[end].text and end not in excluded:
                match = _FLAT_ROW.fullmatch(lines[end].text)
                if not match or _QUANTITY.search(match.group("object")):
                    valid = False
                elif valid:
                    def part(name):
                        return _span(text, lines[end].start + match.start(name), lines[end].start + match.end(name))
                    rows.append(TableRelation(part("object"), lines[header], part("value"), part("unit"), lines[end]))
                end += 1
            if valid and len(rows) >= 2:
                table = TableRelations("captioned_numeric_rows", _span(text, lines[i].start, lines[end - 1].end),
                                       lines[i], (lines[header],), tuple(rows))
                tables.append(_with_notes(table, _neighbor_notes(text, lines, i, end, excluded)))
                i = end
                continue
        i += 1
    from adapters.table_structure import inspect_explicit_structure
    return inspect_explicit_structure(text, tuple(tables))[0]
