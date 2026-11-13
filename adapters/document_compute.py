"""Source-addressed field projection and relational document computation."""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import operator
import re
from typing import Any

from adapters.table_relations import extract_table_relations
from controller.schema import EvidenceClaim, EvidenceUnit, SourceSpan


# Factors use canonical SI units, with time quantities canonicalized to seconds.
UNIT_FACTORS = {
    "": ("", Decimal(1)), "1": ("", Decimal(1)),
    "kg": ("g", Decimal(1000)), "g": ("g", Decimal(1)),
    "mg": ("g", Decimal("0.001")),
    "km": ("m", Decimal(1000)), "m": ("m", Decimal(1)),
    "cm": ("m", Decimal("0.01")),
    "h": ("s", Decimal(3600)), "min": ("s", Decimal(60)),
    "s": ("s", Decimal(1)),
    "美元": ("USD", Decimal(1)), "USD": ("USD", Decimal(1)),
    "元": ("CNY", Decimal(1)), "CNY": ("CNY", Decimal(1)),
    "万元": ("CNY", Decimal(10000)),
    "credit": ("credit", Decimal(1)), "credits": ("credit", Decimal(1)),
    "Credit": ("credit", Decimal(1)), "Credits": ("credit", Decimal(1)),
}


def canonical_quantity(value, unit):
    canonical, factor = UNIT_FACTORS.get(unit, (unit, Decimal(1)))
    literal = str(value)
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", literal):
        literal = literal.replace(",", "")
    return canonical, Decimal(literal) * factor


@dataclass(frozen=True)
class Cell:
    text: str
    source: SourceSpan
    header: SourceSpan | None = None
    unit: str = ""
    qualifiers: tuple[SourceSpan, ...] = ()
    label: str = ""
    qualifier_texts: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentTable:
    table_id: str
    headers: tuple[str, ...]
    header_spans: tuple[SourceSpan | None, ...]
    rows: tuple[tuple[Cell, ...], ...]
    source: SourceSpan
    caption: str = ""
    coverage: tuple[SourceSpan, ...] = ()


@dataclass(frozen=True)
class Field:
    name: str
    column: int
    unit: str = ""
    unit_source: SourceSpan | None = None


@dataclass(frozen=True)
class Projection:
    table_id: str
    fields: tuple[Field, ...]
    predicates: tuple[Predicate, ...] = ()


@dataclass(frozen=True)
class Predicate:
    field: str
    operation: str
    value: Any
    unit: str = ""
    numeric: bool = False

    def __post_init__(self):
        if isinstance(self.value, (int, float, Decimal)):
            object.__setattr__(self, "value", str(self.value))
            object.__setattr__(self, "numeric", True)


@dataclass(frozen=True)
class Join:
    right_index: int
    left_fields: tuple[str, ...]
    right_fields: tuple[str, ...]


@dataclass(frozen=True)
class DocumentProgram:
    need_id: str
    projections: tuple[Projection, ...]
    population_table_ids: tuple[str, ...]
    operation: str = "lookup"
    value_field: str = ""
    entity_fields: tuple[str, ...] = ()
    predicates: tuple[Predicate, ...] = ()
    joins: tuple[Join, ...] = ()
    group_by: tuple[str, ...] = ()
    distinct: tuple[str, ...] = ()
    condition_ids: tuple[str, ...] = ()
    unresolved_conditions: tuple[str, ...] = ()
    excluded_table_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentResult:
    claims: tuple[EvidenceClaim, ...]
    conclusions: tuple[EvidenceClaim, ...]
    covered_tables: tuple[str, ...]
    source_spans: tuple[SourceSpan, ...]
    complete: bool


@dataclass(frozen=True)
class _Record:
    fields: dict[str, Cell]
    sources: tuple[SourceSpan, ...]


_COMPARISONS = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt,
                "le": operator.le, "gt": operator.gt, "ge": operator.ge}
_NUMERIC_AGGREGATES = frozenset({"sum", "mean", "min", "max"})
_NUMBER = re.compile(r"[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?")


def quantity(cell):
    """Read a numeric cell with its explicit unit; preserve its literal source."""
    text = cell.text.strip()
    negative = text.startswith("(") and text.endswith(")")
    literal = text[1:-1].strip() if negative else text
    if cell.unit and literal.endswith(cell.unit):
        literal = literal[:-len(cell.unit)].strip()
    if not _NUMBER.fullmatch(literal):
        raise ValueError(f"cell is not a numeric value: {cell.text!r}")
    unit, value = canonical_quantity(literal, cell.unit)
    return unit, -value if negative else value


def _literal_unit(relation):
    if relation.unit is not None:
        return relation.unit.text
    header = re.search(r"[\[(]([^\])]+)[\])]$", relation.column.text)
    if header and header[1] in UNIT_FACTORS:
        return header[1]
    for unit in sorted(UNIT_FACTORS, key=len, reverse=True):
        if unit and unit != "1" and relation.value.text.endswith(unit):
            number = relation.value.text[:-len(unit)].strip()
            if _NUMBER.fullmatch(number):
                return unit
    return ""


def _span(block, start, end):
    base = block.span[0] if block.span else 0
    key = block.key
    identity = f"{key.corpus_id}:{key.document_id}:{key.passage_id}:{key.version}:{base+start}:{base+end}"
    return SourceSpan(identity, key, base+start, base+end)


def _unique_spans(spans):
    return tuple({(s.source_key, s.start, s.end): s for s in spans}.values())


def _cell_sources(cell):
    return (cell.source, *((cell.header,) if cell.header else ()), *cell.qualifiers)


def _record_sources(record):
    return record.sources


def _record_key(record, fields):
    return tuple(record.fields[name].text.strip() for name in fields)


def _matches(record, predicate):
    cell = record.fields[predicate.field]
    if predicate.operation == "contains":
        return str(predicate.value) in cell.text
    if predicate.operation == "in":
        return cell.text.strip() in predicate.value
    if predicate.operation == "not_in":
        return cell.text.strip() not in predicate.value
    if predicate.numeric:
        unit, actual = quantity(cell)
        expected_unit, expected = canonical_quantity(predicate.value, predicate.unit or cell.unit)
        if unit != expected_unit:
            raise ValueError(f"predicate unit {predicate.unit!r} ({expected_unit}) differs from column unit {cell.unit!r} ({unit})")
    else:
        actual, expected = cell.text.strip(), str(predicate.value).strip()
    return _COMPARISONS[predicate.operation](actual, expected)


class DocumentIndex:
    def __init__(self, tables, blocks=()):
        self.tables = {table.table_id: table for table in tables}
        self.blocks = tuple(blocks)

    @classmethod
    def from_blocks(cls, blocks):
        grouped = {}
        for block in blocks:
            grouped.setdefault(block.key, []).append(block)
        blocks = []
        for windows in grouped.values():
            merged = []
            for block in sorted(windows, key=lambda item: (item.span or (0, len(item.text)))[0]):
                start, end = block.span or (0, len(block.text))
                if merged and start <= merged[-1].span[1]:
                    previous = merged[-1]
                    merged[-1] = replace(previous, text=previous.text+block.text[max(0, previous.span[1]-start):],
                                         span=(previous.span[0], max(previous.span[1], end)), token_ids=())
                else:
                    merged.append(replace(block, span=(start, end)))
            blocks.extend(merged)
        blocks = tuple(blocks)
        tables = []
        for block in blocks:
            for table in extract_table_relations(block.text):
                source = _span(block, table.span.start, table.span.end)
                table_id = f"table:{source.span_id}"
                headers = tuple(header.text for header in table.headers)
                header_spans = tuple(_span(block, h.start, h.end) for h in table.headers)
                columns = {(h.start, h.end): index for index, h in enumerate(table.headers)}
                flat = table.syntax != "markdown"
                if flat:
                    headers = ("", *headers)
                    header_spans = (None, *header_spans)
                rows = {}
                for relation in table.relations:
                    row_key = relation.row_span.start, relation.row_span.end
                    notes = (*relation.parent_path, *relation.notes, *table.source_notes)
                    if table.caption:
                        notes = (table.caption, *notes)
                    qualifiers = tuple(_span(block, s.start, s.end) for s in notes)
                    qualifier_texts = tuple(s.text for s in notes)
                    row = rows.setdefault(row_key, [None] * len(headers))
                    row[0] = Cell(relation.object.text,
                        _span(block, relation.object.start, relation.object.end),
                        None if flat else header_spans[0], qualifiers=qualifiers,
                        label=headers[0], qualifier_texts=qualifier_texts)
                    column = columns[relation.column.start, relation.column.end] + int(flat)
                    row[column] = Cell(relation.value.text,
                        _span(block, relation.value.start, relation.value.end),
                        _span(block, relation.column.start, relation.column.end),
                        _literal_unit(relation), qualifiers, relation.column.text, qualifier_texts)
                tables.append(DocumentTable(table_id, headers, header_spans,
                    tuple(tuple(row) for row in rows.values()), source,
                    table.caption.text if table.caption else block.title))
        return cls(tables, blocks)

    def project(self, projection, required_fields=()):
        table = self.tables[projection.table_id]
        required = {*required_fields, *(predicate.field for predicate in projection.predicates)}
        records = []
        for row in table.rows:
            fields = {name: cell for name, cell in zip(table.headers, row) if name in required}
            fields.update({field.name: (replace(row[field.column], unit=field.unit,
                qualifiers=(*row[field.column].qualifiers, field.unit_source))
                if field.unit_source is not None and not row[field.column].unit else row[field.column])
                for field in projection.fields})
            sources = _unique_spans(span for cell in fields.values() for span in _cell_sources(cell))
            record = _Record(fields, sources)
            if all(_matches(record, predicate) for predicate in projection.predicates):
                records.append(record)
        return tuple(records)

    def outside_tables(self, units):
        output, replacements = [], {}
        table_ranges = {}
        for table in self.tables.values():
            table_ranges.setdefault(table.source.source_key, set()).add((table.source.start, table.source.end))
            for row in table.rows:
                for cell in row:
                    for source in cell.qualifiers:
                        table_ranges.setdefault(source.source_key, set()).add((source.start, source.end))
        for unit in units:
            source = unit.source_spans[0]
            parts = [(source.start, source.end)]
            for left, right in sorted(table_ranges.get(source.source_key, ())):
                parts = [(start, end) for first, last in parts
                         for start, end in ((first, min(last, left)), (max(first, right), last)) if start < end]
            replacements[unit.unit_id] = []
            for start, end in parts:
                text = unit.text[start-source.start:end-source.start]
                if text.strip():
                    identity = f"{unit.unit_id}:text:{start}:{end}"
                    replacements[unit.unit_id].append(identity)
                    span = SourceSpan(identity, source.source_key, start, end)
                    output.append(replace(unit, unit_id=identity, source_spans=(span,), text=text))
        return tuple(replace(unit, dependencies=tuple(identity for dependency in unit.dependencies
            for identity in replacements.get(dependency, ()))) for unit in output)

    def binding_units(self, sample_rows=2):
        """Encode headers and a small literal row sample; compute over all rows later."""
        units = []
        for table in self.tables.values():
            block = next(block for block in self.blocks if block.key == table.source.source_key
                         and (block.span or (0, len(block.text)))[0] <= table.source.start
                         and table.source.end <= (block.span or (0, len(block.text)))[1])
            base = block.span[0] if block.span else 0
            spans = [span for span in table.header_spans if span is not None]
            spans.extend(cell.source for row in table.rows[:sample_rows] for cell in row)
            end = max(span.end for span in spans)
            source = SourceSpan(f"{table.table_id}:binding", block.key, table.source.start, end)
            text = block.text[source.start-base:source.end-base]
            units.append(EvidenceUnit(source.span_id, (source,), text, table.caption, table.table_id))
            qualifiers = _unique_spans(span for row in table.rows for cell in row for span in cell.qualifiers)
            for span in qualifiers:
                if source.start <= span.start and span.end <= source.end:
                    continue
                literal = block.text[span.start-base:span.end-base]
                identity = f"{table.table_id}:qualifier:{span.start}:{span.end}"
                units.append(EvidenceUnit(identity, (span,), literal, table.caption, table.table_id))
        return tuple(units)

    def records(self, program):
        """Materialize only the projected, joined and selected fields."""
        required = {program.value_field, *program.entity_fields, *program.group_by, *program.distinct,
                    *(predicate.field for predicate in program.predicates),
                    *(name for join in program.joins for name in (*join.left_fields, *join.right_fields))}
        projected = tuple(self.project(projection, required) for projection in program.projections)
        if program.joins:
            records = projected[0]
            for join in program.joins:
                right = {}
                for record in projected[join.right_index]:
                    right.setdefault(_record_key(record, join.right_fields), []).append(record)
                records = tuple(_Record({**left.fields, **match.fields},
                    _unique_spans((*left.sources, *match.sources))) for left in records
                    for match in right.get(_record_key(left, join.left_fields), ()))
        else:
            records = tuple(record for group in projected for record in group)
        records = tuple(record for record in records if all(_matches(record, p) for p in program.predicates))
        if program.distinct:
            unique = {}
            for record in records:
                key = _record_key(record, program.distinct)
                if key in unique:
                    previous = unique[key]
                    if program.value_field:
                        left, right = previous.fields[program.value_field], record.fields[program.value_field]
                        values = ((quantity(left), quantity(right)) if program.operation in _NUMERIC_AGGREGATES
                                  else (left.text, right.text))
                        if values[0] != values[1]:
                            raise ValueError("distinct entity has conflicting measure values")
                    record = _Record(previous.fields, _unique_spans((*previous.sources, *record.sources)))
                unique[key] = record
            records = tuple(unique.values())
        return records

    def execute(self, program):
        """Aggregate the complete bound population, retaining source lineage."""
        records = self.records(program)
        used = (0, *(join.right_index for join in program.joins)) if program.joins else range(len(program.projections))
        covered = tuple(dict.fromkeys(program.projections[index].table_id for index in used))
        scope = _unique_spans(span for table_id in dict.fromkeys((*covered, *program.excluded_table_ids))
            for span in (self.tables[table_id].coverage or (self.tables[table_id].source,)))
        complete = (set(program.population_table_ids) <= set(covered) | set(program.excluded_table_ids)
                    and not program.unresolved_conditions)
        claims = []
        for index, record in enumerate(records):
            entity = " / ".join(_record_key(record, program.entity_fields)) if program.entity_fields else str(index)
            sources = _record_sources(record)
            value = record.fields[program.value_field].text if program.value_field else {name: c.text for name, c in record.fields.items()}
            unit = record.fields[program.value_field].unit if program.value_field else ""
            claims.append(EvidenceClaim(f"document:{program.need_id}:{index}", program.need_id,
                "extract", text="; ".join(f"{name}={c.text}" for name, c in record.fields.items()),
                value=value, unit=unit, conditions=program.condition_ids,
                source_spans=sources, operation="project", entity=entity, relation=program.value_field,
                version=" / ".join(dict.fromkeys(s.source_key.version for s in sources)),
                attributes=tuple((name, cell.label, cell.text, cell.unit) for name, cell in record.fields.items()
                                 if name not in program.entity_fields),
                qualifiers=tuple(dict.fromkeys(text for cell in record.fields.values()
                                               for text in cell.qualifier_texts))))
        if program.operation == "lookup" or not complete:
            return DocumentResult(tuple(claims), tuple(claims), covered, scope, complete)
        groups = {}
        for index, record in enumerate(records):
            groups.setdefault(_record_key(record, program.group_by), []).append(index)
        if not groups and not program.group_by:
            groups[()] = []
        conclusions = []
        for group_index, (group, indices) in enumerate(groups.items()):
            selected = [records[i] for i in indices]
            unit = ""
            if program.operation == "count":
                value = len(selected)
            elif program.operation in _NUMERIC_AGGREGATES:
                quantities = [quantity(record.fields[program.value_field]) for record in selected]
                units = {unit for unit, _ in quantities}
                if len(units) > 1:
                    raise ValueError("aggregation requires compatible units")
                unit = next(iter(units), "")
                numbers = [number for _, number in quantities]
                if program.operation == "sum":
                    value = sum(numbers, Decimal(0))
                elif program.operation == "mean":
                    value = sum(numbers, Decimal(0)) / len(numbers) if numbers else None
                else:
                    value = {"min": min, "max": max}[program.operation](numbers) if numbers else None
                value = str(value) if value is not None else None
            elif program.operation == "group":
                value = [{name: cell.text for name, cell in record.fields.items()} for record in selected]
            else:
                raise ValueError(f"unknown document operation: {program.operation}")
            parents = tuple(claims[i].claim_id for i in indices)
            spans = _unique_spans(span for i in indices for span in claims[i].source_spans) or scope
            labels = tuple(dict.fromkeys(record.fields[program.value_field].label for record in selected
                                          if program.value_field and record.fields[program.value_field].label))
            conclusions.append(EvidenceClaim(f"document:{program.need_id}:result:{group_index}",
                program.need_id, "derived", text=str(value), value=value, unit=unit,
                conditions=program.condition_ids, source_spans=spans, operation=program.operation,
                parent_claim_ids=parents, entity=" / ".join(group), relation=" / ".join(labels) or program.value_field,
                qualifiers=tuple(dict.fromkeys(text for record in selected for cell in record.fields.values()
                                               for text in cell.qualifier_texts))))
        return DocumentResult(tuple(claims), tuple(conclusions), covered, scope, complete)


def _binding_columns(table, binding):
    span = binding.source_span
    if span.source_key != table.source.source_key:
        return ()
    headers = tuple((column, None) for column, header in enumerate(table.header_spans)
        if header is not None and header.start < span.end and span.start < header.end)
    if headers:
        return headers
    return tuple((column, row_index) for row_index, row in enumerate(table.rows)
        for column, cell in enumerate(row) if cell.source.start < span.end and span.start < cell.source.end)


def _condition_predicate(query, condition, field):
    operation = query.filter_operators[condition.condition_id]
    values = tuple(query.text((ref,)) for ref in condition.value_spans)
    if operation == "in":
        return Predicate(field, operation, values)
    value = " ".join(values)
    unit = ""
    if operation in {"lt", "le", "gt", "ge"}:
        if _NUMBER.fullmatch(value):
            value = Decimal(value.replace(",", ""))
        else:
            for name in sorted(UNIT_FACTORS, key=len, reverse=True):
                if name and name != "1" and value.endswith(name):
                    literal = value[:-len(name)].strip()
                    if _NUMBER.fullmatch(literal):
                        value, unit = Decimal(literal.replace(",", "")), name
                        break
    return Predicate(field, operation, value, unit)


def build_programs(query, evidence, index):
    """Compile learned source bindings into explicit document operations."""
    programs = []
    conditions = {condition.condition_id: condition for condition in query.task.conditions}
    for need in query.task.needs:
        if need.operation not in {"count", "group", *_NUMERIC_AGGREGATES}:
            continue
        bindings = [binding for binding in evidence.bindings if binding.need_id == need.need_id]
        joining = need.need_id in query.join_needs
        projections, mapped_conditions, entities, grouping, join_fields = [], None, (), (), []
        for table in index.tables.values():
            located = [(binding, column, row) for binding in bindings
                       for column, row in _binding_columns(table, binding)]
            if not located:
                continue
            roles = {}
            for binding, column, row in sorted(located, key=lambda item: -item[0].score):
                roles.setdefault(binding.role, {}).setdefault(column, (binding, row))
            fields, local_predicates, keys, local_conditions = {}, [], {}, set()
            measures = roles.get("measure", {})
            period = roles.get("period", {})
            row_measure = next(((column, item) for column, item in measures.items()
                                if column == 0 and item[1] is not None), None)
            if row_measure:
                _, (_, row_index) = row_measure
                fields["row"] = 0
                local_predicates.append(Predicate("row", "eq", table.rows[row_index][0].text))
                measure_column = next((column for column in period if column != 0), None)
            else:
                measure_column = next(iter(measures), None)
            if need.operation in _NUMERIC_AGGREGATES and measure_column is None and not joining:
                continue
            if measure_column is not None:
                fields["measure"] = measure_column
            entity_names = []
            for position, (column, (binding, _)) in enumerate(roles.get("entity", {}).items()):
                name = f"entity:{position}"
                fields[name] = column
                entity_names.append(name)
                if binding.relation_group:
                    key = f"join:{binding.relation_group}"
                    fields[key] = column
                    keys[binding.relation_group] = key
            group_names = []
            for position, column in enumerate(roles.get("group", {})):
                name = f"group:{position}"
                fields[name] = column
                group_names.append(name)
            for binding, column, row in located:
                condition_id = binding.condition_id
                if binding.role not in {"filter", "period"} or not condition_id:
                    continue
                condition = conditions[condition_id]
                if row_measure and binding.role == "period" and row is None:
                    local_conditions.add(condition_id)
                    continue
                name = f"condition:{condition_id}"
                if name not in fields:
                    fields[name] = column
                    local_predicates.append(_condition_predicate(query, condition, name))
                    local_conditions.add(condition_id)
            if not fields:
                continue
            unit_sources = (*[span for span in table.header_spans if span is not None],
                            *(span for row in table.rows for cell in row for span in cell.qualifiers))
            units = [binding for binding in bindings if binding.role == "unit" and any(
                source.source_key == binding.source_span.source_key
                and source.start <= binding.source_span.start and binding.source_span.end <= source.end
                for source in unit_sources)]
            declared_unit = max(units, key=lambda binding: binding.score) if units else None
            projections.append(Projection(table.table_id,
                tuple(Field(name, column, declared_unit.text, declared_unit.source_span)
                    if name == "measure" and declared_unit is not None else Field(name, column)
                    for name, column in fields.items()), tuple(local_predicates)))
            join_fields.append(keys)
            mapped_conditions = (local_conditions if mapped_conditions is None else
                mapped_conditions | local_conditions if joining else mapped_conditions & local_conditions)
            entities = tuple(entity_names) or entities
            grouping = tuple(group_names) or grouping
        if not projections:
            continue
        joins = []
        if joining:
            available = dict(join_fields[0])
            remaining = set(range(1, len(projections)))
            while remaining:
                connected = next((position for position in sorted(remaining)
                                  if available.keys() & join_fields[position].keys()), None)
                if connected is None:
                    break
                shared = tuple(sorted(available.keys() & join_fields[connected].keys()))
                joins.append(Join(connected, tuple(available[key] for key in shared),
                                  tuple(join_fields[connected][key] for key in shared)))
                available.update(join_fields[connected])
                remaining.remove(connected)
            if not joins:
                continue
        if need.operation == "group" and not grouping:
            continue
        population = tuple(index.tables) if "all_supplied_documents" in (
            query.task.coverage_scope, need.scope) else tuple(
            projection.table_id for projection in projections)
        unresolved = tuple(identity for identity in need.condition_ids
            if conditions[identity].relation_type != "VERSION" and identity not in mapped_conditions)
        programs.append(DocumentProgram(need.need_id, tuple(projections), population,
            operation=need.operation, value_field="measure" if need.operation in _NUMERIC_AGGREGATES else "",
            entity_fields=entities, joins=tuple(joins), group_by=grouping,
            distinct=entities if need.need_id in query.distinct_needs else (),
            condition_ids=need.condition_ids, unresolved_conditions=unresolved))
    return tuple(programs)


@dataclass(frozen=True)
class SemanticBinding:
    blocks: tuple
    instruction: str
    schema: dict
    decode: object


def _query_description(query, need_ids):
    ids = set(need_ids)
    selected = tuple(need for need in query.task.needs if need.need_id in ids)
    condition_ids = {identity for need in selected for identity in need.condition_ids}
    return {"question": query.task.raw_question, "proposed_needs": [{"id": need.need_id,
        "subject": query.text(need.subject_refs),
        "request": query.text(need.anchor_spans), "conditions": need.condition_ids} for need in selected], "proposed_conditions": [{
            "id": condition.condition_id, "type": condition.relation_type,
            "field": query.text((condition.key_span,)) if condition.key_span else "",
            "values": [query.text((ref,)) for ref in condition.value_spans],
            "predicate": query.text((condition.predicate_span,)) if condition.predicate_span else "",
            "operator": query.filter_operators[condition.condition_id]}
            for condition in query.task.conditions if condition.condition_id in condition_ids]}


def _word_source(unit, address):
    from controller.source_windows import numbered_text
    _, offsets = numbered_text(unit.text)
    first, last = sorted(address[1:])
    start, cursor = offsets[first][0], offsets[last][1]
    origin = unit.source_spans[0]
    span = SourceSpan(f"{origin.span_id}:{start}:{cursor}", origin.source_key,
                      origin.start + start, origin.start + cursor)
    return unit.text[start:cursor], span


def _word_address_schema(units):
    from controller.source_windows import numbered_text
    return {"oneOf": [{"type": "array", "prefixItems": [{"const": f"S{i+1}"},
        *[{"type": "integer", "minimum": 0, "maximum": len(offsets)-1} for _ in range(2)]],
        "minItems": 3, "items": False}
        for i, unit in enumerate(units) if (offsets := numbered_text(unit.text)[1])]}


def _word_blocks(units):
    from dataclasses import replace
    from controller.schema import EvidenceBlock
    from controller.source_windows import numbered_text
    return tuple(EvidenceBlock(replace(unit.source_spans[0].source_key,
        passage_id=f"{unit.source_spans[0].source_key.passage_id}:indexed-words:{unit.source_spans[0].start}:{unit.source_spans[0].end}"),
        "", unit.heading, numbered_text(unit.text)[0]) for unit in units)


def semantic_binding_schema(index, need_ids, operations=None):
    def array(item):
        return {"type": "array", "items": item}
    def obj(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}
    string = {"type": "string"}
    predicate = obj({"field": string, "operation": {"enum": [*list(_COMPARISONS), "in", "contains"]},
        "value": {"anyOf": [string, {"type": "number"}, array(string)]}, "unit": string})
    table_ids = [f"T{i+1}" for i in range(len(index.tables))]
    projections = []
    for identity, table in zip(table_ids, index.tables.values()):
        column = {"type": "integer", "minimum": 0, "maximum": len(table.headers)-1}
        local_predicate = obj({"column": column, **{name: value for name, value in predicate["properties"].items()
                                                  if name != "field"}})
        projections.append(obj({"table_id": {"const": identity}, "fields": array(obj({
            "name": string, "column": column,
            "unit_span": {"anyOf": [{"type": "null"}, _word_address_schema(index.binding_units())]}})),
            "predicates": array(local_predicate)}))
    projection = {"oneOf": projections}
    programs = {identity: obj({
        "operation": ({"const": operations[identity]} if operations and identity in operations else
                      {"enum": ["lookup", "group", "count", "sum", "mean", "min", "max"]}),
        "projections": array(projection), "value_field": string, "entity_fields": array(string),
        "predicates": array(predicate), "joins": array(obj({
            "right_index": {"type": "integer", "minimum": 1},
            "left_fields": array(string), "right_fields": array(string)})),
        "group_by": array(string), "distinct": array(string), "unresolved_conditions": array(string),
        "excluded_table_ids": array({"enum": table_ids})}) for identity in need_ids}
    return obj({"programs": obj(programs)})


def decode_programs(text, query, index):
    import json
    needs = {need.need_id: need for need in query.task.needs}
    result = []
    fragments = {f"S{i+1}": unit for i, unit in enumerate(index.binding_units())}
    tables = {f"T{i+1}": identity for i, identity in enumerate(index.tables)}
    def field_binding(field):
        address = field["unit_span"]
        if address is None:
            return Field(field["name"], field["column"])
        text, span = _word_source(fragments[address[0]], address)
        return Field(field["name"], field["column"], text, span)
    def projection_binding(projection):
        columns = tuple(dict.fromkeys(predicate["column"] for predicate in projection["predicates"]))
        fields = (*[field_binding(field) for field in projection["fields"]],
                  *(Field(f"column:{column}", column) for column in columns))
        predicates = tuple(Predicate(field=f"column:{predicate['column']}",
            **{name: value for name, value in predicate.items() if name != "column"})
            for predicate in projection["predicates"])
        return Projection(tables[projection["table_id"]], fields, predicates)
    for identity, item in json.loads(text)["programs"].items():
        need = needs[identity]
        projections = tuple(projection_binding(projection) for projection in item["projections"])
        population = tuple(index.tables)
        result.append(DocumentProgram(need.need_id, projections, population, item["operation"],
            item["value_field"], tuple(item["entity_fields"]),
            tuple(Predicate(**predicate) for predicate in item["predicates"]),
            tuple(Join(join["right_index"], tuple(join["left_fields"]), tuple(join["right_fields"]))
                  for join in item["joins"]), tuple(item["group_by"]), tuple(item["distinct"]),
            need.condition_ids, tuple(item["unresolved_conditions"]),
            tuple(tables[identity] for identity in item["excluded_table_ids"])))
    return tuple(result)


def semantic_binding(query, index, need_ids, *, operations=None):
    """Bind a document program from schema and examples, without reading all rows."""
    import json
    units = index.binding_units()
    blocks = _word_blocks(units)
    table_ids = {identity: f"T{i+1}" for i, identity in enumerate(index.tables)}
    catalogue = [{"table_id": table_ids[table.table_id], "columns": [
        {"column": column, "header": header} for column, header in enumerate(table.headers)],
        "caption": table.caption, "rows": len(table.rows), "source_start": table.source.start} for table in index.tables.values()]
    instruction = ("Bind the requested fields and conditions to the supplied table schemas. "
        "The original question is authoritative. Auxiliary Need and condition fragments can be incomplete or "
        "incorrect; resolve their intended meaning from the original question rather than imposing their text. "
        "Choose the operation actually requested by the full question. "
        "Use lookup to return fields for semantic classification rather than forcing an arithmetic aggregate. "
        "Samples illustrate columns; the CPU will execute over every row of each selected table. "
        "Return projections and typed filters, explicit equality joins, and requested entity deduplication. "
        "Projection filters address zero-based source column indices, including columns not in the output. "
        "Global filters, joins, value_field, entity_fields, group_by and distinct refer to field names declared "
        "in the projections. Optional unit_span is [source_id,first_word,last_word], with inclusive "
        "displayed word indices; "
        "select only the literal unit in the supplied source. Same-shaped tables use the same field aliases for union; "
        "join fields identify the actual key correspondence. Include all tables in the requested population. "
        "For transposed tables select the period column and filter the requested metric row. "
        "Put explicitly unrelated tables in excluded_table_ids; leave undecided tables unprojected and unexcluded. "
        "Keep conditions that cannot be bound in unresolved_conditions. Do not calculate totals.\n"
        + json.dumps({"task": _query_description(query, need_ids), "tables": catalogue,
                         "sources": [{"unit_id": f"S{i+1}", "table_id": table_ids[unit.table_id]} for i, unit in enumerate(units)]},
                     ensure_ascii=False, separators=(",", ":")))
    return SemanticBinding(blocks, instruction, semantic_binding_schema(index, need_ids, operations),
                           lambda text: decode_programs(text, query, index))


@dataclass(frozen=True)
class SemanticProjection:
    index: DocumentIndex
    programs: tuple[DocumentProgram, ...]
    operations: dict


def semantic_projection_schema(units, need_ids, operations=None):
    def array(item):
        return {"type": "array", "items": item}
    def obj(properties):
        return {"type": "object", "properties": properties, "required": list(properties),
                "additionalProperties": False}
    unit_ids = [f"S{i+1}" for i in range(len(units))]
    address = _word_address_schema(units)
    entity = {"oneOf": [address, {"type": "array", "prefixItems": [{"enum": unit_ids}],
                                 "minItems": 1, "items": False}]}
    optional_address = {"anyOf": [address, {"type": "null"}]}
    return obj({"operations": obj({identity: ({"const": operations[identity]} if operations and identity in operations else
                                  {"enum": ["lookup", "group", "count", "sum", "mean", "min", "max"]})
                                  for identity in need_ids}),
        "unresolved_need_ids": array({"enum": list(need_ids)}),
        "records": array(obj({"need_ids": array({"enum": list(need_ids)}), "entity": entity,
            "measure": optional_address, "unit": optional_address,
            "group": array(address), "qualifiers": array(address)}))})


def decode_projection(text, query, units, need_ids, population_unit_ids):
    import json
    from controller.schema import EvidenceBlock
    data = json.loads(text)
    units_by_id = {f"S{index+1}": unit for index, unit in enumerate(units)}

    def source(address):
        unit = units_by_id[address[0]]
        return _word_source(unit, address)

    tables, programs = [], []
    needs = {need.need_id: need for need in query.task.needs}
    for need_id in need_ids:
        need = needs[need_id]
        operation = data["operations"][need_id]
        records = [record for record in data["records"] if need_id in record["need_ids"]]
        group_count = max((len(record["group"]) for record in records), default=0)
        headers = ("entity", "measure", *(f"group:{i}" for i in range(group_count)))
        grouped = {identity: [] for identity in units_by_id}
        for record in records:
            if len(record["entity"]) == 1:
                unit = units_by_id[record["entity"][0]]
                entity_text, entity_source = unit.source_spans[0].source_key.document_id, unit.source_spans[0]
            else:
                entity_text, entity_source = source(record["entity"])
            qualified = tuple(source(address) for address in record["qualifiers"])
            qualifier_texts = tuple(text for text, _ in qualified)
            qualifiers = tuple(span for _, span in qualified)
            unit_text, unit_source = source(record["unit"]) if record["unit"] else ("", None)
            if unit_source is not None:
                qualifiers = (*qualifiers, unit_source)
            entity = Cell(entity_text, entity_source, qualifiers=qualifiers, qualifier_texts=qualifier_texts)
            measure_text, measure_source = source(record["measure"]) if record["measure"] else ("", entity_source)
            measure = Cell(measure_text, measure_source, unit=unit_text, qualifiers=qualifiers,
                           qualifier_texts=qualifier_texts)
            groups = tuple(Cell(*source(address), qualifiers=qualifiers, qualifier_texts=qualifier_texts)
                           for address in record["group"])
            grouped[record["entity"][0]].append((entity, measure, *groups))
        projections = []
        for identity, rows in grouped.items():
            origin = units_by_id[identity].source_spans[0]
            table_id = f"projection:{need_id}:{units_by_id[identity].unit_id}"
            tables.append(DocumentTable(table_id, headers, tuple(None for _ in headers), tuple(rows), origin))
            projections.append(Projection(table_id, tuple(Field(name, column) for column, name in enumerate(headers))))
        programs.append(DocumentProgram(need_id, tuple(projections),
            tuple(f"projection:{need_id}:{identity}" for identity in population_unit_ids),
            operation=operation, value_field="measure" if any(record["measure"] is not None for record in records) else "",
            entity_fields=("entity",), group_by=tuple(headers[2:]),
            distinct=("entity",) if need_id in query.distinct_needs else (),
            condition_ids=need.condition_ids,
            unresolved_conditions=(need_id,) if need_id in data["unresolved_need_ids"] else ()))
    blocks = tuple(EvidenceBlock(unit.source_spans[0].source_key, "", unit.heading, unit.text,
        span=(unit.source_spans[0].start, unit.source_spans[0].end)) for unit in units)
    return SemanticProjection(DocumentIndex(tables, blocks), tuple(programs), data["operations"])


def semantic_projection(query, units, need_ids, *, population_unit_ids=None, operations=None):
    """Read only requested fields and eligibility once; use CPU aggregation afterwards."""
    import json
    units, need_ids = tuple(units), tuple(need_ids)
    population = tuple(population_unit_ids) if population_unit_ids is not None else tuple(unit.unit_id for unit in units)
    blocks = _word_blocks(units)
    instruction = ("Project only the requested document fields. For each qualifying record, return its entity, "
        "requested measure, literal unit, grouping fields, and the period/condition qualifiers as exact source "
        "references [source_id,first_word,last_word] using inclusive displayed word indices. "
        "The CPU resolves the original text and character spans. "
        "The original question is authoritative. Auxiliary Need and condition fragments can be incomplete or "
        "incorrect; resolve their intended meaning from the original question. An incomplete auxiliary fragment "
        "is not unresolved source information. Choose the operation requested by the full question; use lookup to "
        "return fields for classification or semantic comparison instead of forcing an arithmetic aggregate. "
        "When the requested entity is the document itself, use [source_id]; its identifier is shown "
        "below and the CPU binds that document identity. Otherwise use original word indices for the entity. "
        "A record can serve several Need IDs: extract it once and list all applicable IDs. "
        "Apply the original question's entity, period and conditions in this same pass. Include the measure's "
        "field label, period and conditions in qualifiers. Retain units and qualifiers "
        "from source headings or notes. Do not produce totals, summaries or a second selection list. "
        "Examine each source completely. No matching field in a window is an empty projection; other windows "
        "can supply it. Follow the question's missing-data policy. Report unresolved Need IDs for ambiguous "
        "existing values or applicability, not merely because one window contains no matching field.\n"
        + json.dumps({"task": _query_description(query, need_ids),
            "sources": [{"unit_id": f"S{index+1}", "title": unit.heading,
                         "document": unit.source_spans[0].source_key.document_id}
                        for index, unit in enumerate(units)]}, ensure_ascii=False, separators=(",", ":")))
    return SemanticBinding(blocks, instruction,
        semantic_projection_schema(units, need_ids, operations),
        lambda text: decode_projection(text, query, units, need_ids, population))


def semantic_evidence_projection(question, units, *, retained_tokens, tokenizer, source_headers=()):
    """Select original line indices; the model never reconstructs source text."""
    import json
    from dataclasses import replace
    from controller.schema import EvidenceBlock
    from controller.source_windows import numbered_lines
    identities, blocks, locations, properties = {}, [], {}, {}
    for index, unit in enumerate(units):
        identity = f"S{index+1}"
        identities[identity] = unit
        body, offsets = numbered_lines(unit.text, tokenizer)
        locations[identity] = offsets
        origin = unit.source_spans[0]
        key = replace(origin.source_key,
            passage_id=f"{origin.source_key.passage_id}:indexed-lines:{origin.start}:{origin.end}")
        blocks.append(EvidenceBlock(key, "", unit.heading, body))
        properties[identity] = {"type": "array", "maxItems": len(offsets),
            "items": {"type": "integer", "minimum": 0, "maximum": len(offsets)-1}}
    schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    instruction = ("Examine every supplied source. For each source ID return the selected original line numbers "
        "containing the fields and statements needed to answer the question. Use an empty list for unrelated sources. "
        "Preserve source entity/title, relevant attributes, directed relationships, conditions, exceptions, dates, "
        "units and versions. Retain facts needed to decide eligibility even when they are negative or ambiguous; "
        "the final answer decides which members satisfy the conditions. For global relationships keep relevant "
        "cross-source links even when the other source is in another batch. The document catalogue identifies "
        "the supplied population. For relationships within that population, retain links to its members, "
        "not unrelated external entities. Return each selected line number once, "
        "in original order. Return only source IDs and arrays of line numbers, without copied text or explanations. "
        f"The selected original lines share a budget of {retained_tokens} tokens; their token counts are shown. "
        "Keep necessary facts and leave unrelated lines out.\nQuestion: " + question +
        "\nDocument catalogue (literal openings, not complete evidence):\n" +
        "\n".join(f"{block.title}\n{block.text}" for block in source_headers))

    def decode_evidence(text):
        selected = json.loads(text)
        result = []
        for identity, unit in identities.items():
            offsets, merged = locations[identity], []
            for number in sorted(set(selected[identity])):
                left, right = offsets[number]
                if merged and left == merged[-1][1]:
                    merged[-1] = merged[-1][0], right
                else:
                    merged.append((left, right))
            origin = unit.source_spans[0]
            result.extend(EvidenceBlock(origin.source_key, "", unit.heading, unit.text[left:right],
                span=(origin.start+left, origin.start+right)) for left, right in merged)
        return tuple(result)

    return SemanticBinding(tuple(blocks), instruction, schema, decode_evidence)

def merge_projections(projections):
    """Combine independently read source partitions before the single CPU aggregation."""
    from dataclasses import replace
    tables, blocks, programs = {}, {}, {}
    for projection in projections:
        tables.update(projection.index.tables)
        blocks.update(((block.key, block.span), block) for block in projection.index.blocks)
        for program in projection.programs:
            previous = programs.get(program.need_id)
            if previous is not None:
                program = replace(program, projections=(*previous.projections, *program.projections),
                    population_table_ids=tuple(dict.fromkeys((*previous.population_table_ids, *program.population_table_ids))),
                    group_by=program.group_by or previous.group_by,
                    unresolved_conditions=tuple(dict.fromkeys((*previous.unresolved_conditions, *program.unresolved_conditions))))
            programs[program.need_id] = program
    return SemanticProjection(DocumentIndex(tables.values(), blocks.values()),
                              tuple(programs.values()), projections[0].operations)


def combine_populations(parts):
    """Normalize table and prose projections, then aggregate each Need once."""
    tables, blocks, grouped = [], {}, {}
    for part_index, (index, program) in enumerate(parts):
        if program.operation == "lookup":
            program = replace(program, group_by=())
        blocks.update(((block.key, block.span), block) for block in index.blocks)
        rows = []
        names = ("entity", "measure", *(f"group:{i}" for i in range(len(program.group_by))))
        for position, record in enumerate(index.records(program)):
            sources = record.sources
            entity = " / ".join(_record_key(record, program.entity_fields)) if program.entity_fields else str(position)
            represented = {*program.entity_fields, program.value_field, *program.group_by}
            context = tuple(dict.fromkeys((
                *(text for cell in record.fields.values() for text in cell.qualifier_texts),
                *(f"{cell.label or name}: {cell.text}" for name, cell in record.fields.items()
                  if name not in represented))))
            row = [Cell(entity, sources[0], qualifiers=sources[1:], qualifier_texts=context)]
            if program.value_field:
                cell = record.fields[program.value_field]
                row.append(cell)
            else:
                row.append(Cell("", sources[0], qualifiers=sources[1:]))
            row.extend(record.fields[name] for name in program.group_by)
            rows.append(tuple(row))
        used = (0, *(join.right_index for join in program.joins)) if program.joins else range(len(program.projections))
        processed = tuple(dict.fromkeys((*(program.projections[i].table_id for i in used),
                                        *program.excluded_table_ids)))
        scope = _unique_spans(span for identity in processed
            for span in (index.tables[identity].coverage or (index.tables[identity].source,)))
        identity = f"population:{program.need_id}:{part_index}"
        if scope:
            tables.append(DocumentTable(identity, names, tuple(None for _ in names), tuple(rows), scope[0], coverage=scope))
        entry = grouped.setdefault(program.need_id, {"source": program, "projections": [], "population": [],
                                                    "unresolved": set()})
        if scope:
            entry["projections"].append(Projection(identity, tuple(Field(name, col) for col, name in enumerate(names))))
            entry["population"].append(identity)
        entry["population"].extend(set(program.population_table_ids) - set(processed))
        entry["unresolved"].update(program.unresolved_conditions)
    programs = []
    for need_id, entry in grouped.items():
        source = entry["source"]
        programs.append(DocumentProgram(need_id, tuple(entry["projections"]), tuple(entry["population"]),
            operation=source.operation, value_field="measure" if source.value_field else "",
            entity_fields=("entity",), group_by=tuple(f"group:{i}" for i in range(len(source.group_by))),
            distinct=("entity",) if source.distinct else (), condition_ids=source.condition_ids,
            unresolved_conditions=tuple(sorted(entry["unresolved"]))))
    return DocumentIndex(tables, blocks.values()), tuple(programs)
