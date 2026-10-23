"""Optional literal structure extension; never recover hierarchy lost in scraping.

Unnumbered tables retain their unsplit header. Parent paths come only from ATX
Markdown headings and genuinely indented colon lists, not nearby prose. The
diagnostics are rejected parse candidates, not table-recall ground truth.
"""
from __future__ import annotations

from dataclasses import replace
import re

from adapters.table_relations import (
    TableRelation, TableRelations, _FLAT_ROW, _QUANTITY, _neighbor_notes, _span, _with_notes,
)

_HEADING = re.compile(r"^(#{1,6}) +\S")
_BULLET = re.compile(r"^( *)(?:[-+*]|\d+[.)]) +(?P<body>\S.*)$")
_PROPERTY = re.compile(
    r"\b(?:size|limit|maximum|minimum|latency|time|rate|capacity|length|duration|"
    r"speed|weight|height|distance|throughput|memory|timeout|quota)\b", re.I)


def _title(text):
    return (bool(text) and len(text.split()) <= 12 and not re.search(r"[.!?;|]", text)
            and not _BULLET.match(text) and not _QUANTITY.search(text))


def _header(text):
    return (_title(text) and len(text.split()) >= 2 and _PROPERTY.search(text)
            and not re.search(r"\b(?:is|are|was|were|has|have|must|shall|see)\b|\d", text, re.I))


def inspect_explicit_structure(text, legacy_tables=()):
    """Return expanded tables and source-local rejection records.

    Numeric rows require one known-unit quantity at the end. A title/header pair
    must be separate lines; no cell boundaries or ordinary-prose conditions are
    guessed. Four-space code remains excluded unless a literal parent bullet
    already establishes that indentation as a nested list.
    """
    lines, raw_lines, excluded, paths = [], [], set(), []
    headings, parents, nested, rejected = [], [], [], []
    offset, fence = 0, None
    for raw in text.splitlines(keepends=True):
        line = _span(text, offset, offset + len(raw))
        index = len(lines)
        lines.append(line)
        raw_lines.append(raw)
        paths.append(tuple(span for _, span in headings))
        offset += len(raw)
        marker = re.match(r"(`{3,}|~{3,})", line.text)
        if fence or marker:
            excluded.add(index)
            parents.clear()
            if fence:
                if marker and marker.group()[0] == fence[0] and len(marker.group()) >= len(fence):
                    fence = None
            else:
                fence = marker.group()
            continue
        if not line.text:
            continue
        bullet = _BULLET.match(raw.rstrip("\r\n"))
        indent = len(bullet.group(1)) if bullet else 0
        nested_bullet = bullet and parents and indent < parents[-1][0] + 4
        if raw.startswith("\t") or (raw.startswith("    ") and not nested_bullet):
            excluded.add(index)
            parents.clear()
            continue
        heading = _HEADING.match(line.text)
        if heading:
            level = len(heading.group(1))
            headings = [(depth, span) for depth, span in headings if depth < level]
            paths[index] = tuple(span for _, span in headings)
            headings.append((level, line))
            parents.clear()
            continue
        if not bullet:
            parents.clear()
            continue
        while parents and indent < parents[-1][0]:
            parents.pop()
        body = bullet.group("body")
        body_start = line.start + len(line.text) - len(body)
        body_span = _span(text, body_start, line.end)
        if body.endswith(":") and not _QUANTITY.search(body):
            parents.append((bullet.start("body"), body_span))
            continue
        match = _FLAT_ROW.fullmatch(body)
        if not match or _QUANTITY.search(match.group("object")):
            continue
        if not parents:
            rejected.append({"span": line, "reason": "numeric_list_without_explicit_indented_parent"})
            continue
        # Named values require the colon present in the original list item.
        object_text = match.group("object")
        if not object_text.endswith(":"):
            rejected.append({"span": line, "reason": "nested_numeric_item_without_name_value_colon"})
            continue
        def part(name):
            return _span(text, body_start + match.start(name), body_start + match.end(name))
        parent_path = (*paths[index], *(span for _, span in parents))
        object_span = _span(text, body_start + match.start("object"), body_start + match.end("object") - 1)
        relation = TableRelation(object_span, parents[-1][1], part("value"), part("unit"),
                                 line, parent_path=parent_path)
        nested.append(TableRelations("nested_numeric_list", line, None, (parents[-1][1],), (relation,)))

    tables = []
    for table in legacy_tables:
        index = next(i for i, line in enumerate(lines) if line.start == table.span.start)
        tables.append(replace(table, relations=tuple(replace(row, parent_path=paths[index]) for row in table.relations)))
    tables.extend(nested)
    covered = [(table.span.start, table.span.end) for table in tables]
    i = 0
    while i < len(lines):
        line = lines[i]
        match = _FLAT_ROW.fullmatch(line.text)
        if (i in excluded or not match or _QUANTITY.search(match.group("object"))
                or _BULLET.match(raw_lines[i].rstrip("\r\n"))
                or any(start <= line.start < end for start, end in covered)):
            i += 1
            continue
        first, rows = i, []
        while i < len(lines) and lines[i].text and i not in excluded:
            match = _FLAT_ROW.fullmatch(lines[i].text)
            if (not match or _QUANTITY.search(match.group("object"))
                    or _BULLET.match(raw_lines[i].rstrip("\r\n"))):
                break
            rows.append((i, match))
            i += 1
        header = first - 1
        while header >= 0 and not lines[header].text:
            header -= 1
        caption = header - 1
        while caption >= 0 and not lines[caption].text:
            caption -= 1
        reason = None
        if len(rows) < 2:
            reason = "fewer_than_two_contiguous_numeric_rows"
        elif (i < len(lines) and lines[i].text and i not in excluded):
            reason = "mixed_or_ragged_numeric_block"
        elif header < 0 or header in excluded or not _header(lines[header].text):
            reason = "no_explicit_quantity_header"
        elif (caption < 0 or caption in excluded or not _title(lines[caption].text)
              or (caption > 0 and lines[caption - 1].text and not _HEADING.match(lines[caption].text))):
            reason = "no_separate_short_caption"
        if reason:
            rejected.append({"span": _span(text, lines[first].start, lines[i - 1].end), "reason": reason})
            continue
        relations = []
        for row_index, match in rows:
            def part(name):
                return _span(text, lines[row_index].start + match.start(name), lines[row_index].start + match.end(name))
            relations.append(TableRelation(part("object"), lines[header], part("value"), part("unit"),
                                           lines[row_index], parent_path=paths[caption]))
        table = TableRelations("unnumbered_numeric_rows", _span(text, lines[caption].start, lines[i - 1].end),
                               lines[caption], (lines[header],), tuple(relations))
        tables.append(_with_notes(table, _neighbor_notes(text, lines, caption, i, excluded)))
    return tuple(sorted(tables, key=lambda t: t.span.start)), tuple(rejected)
