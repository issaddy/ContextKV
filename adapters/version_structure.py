"""Version, section and field correspondence over original source addresses."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from adapters.document_compute import DocumentIndex, _span, _unique_spans


@dataclass(frozen=True)
class VersionPart:
    address: tuple[str, ...]
    text: str
    spans: tuple


@dataclass(frozen=True)
class VersionChange:
    family: str
    before_version: str
    after_version: str
    address: tuple[str, ...]
    before: tuple
    after: tuple
    kind: str


def _sections(block, metadata):
    headings = list(re.finditer(r"^(#{1,6})\s+(.+?)\s*#*\s*$", block.text, re.MULTILINE))
    if not headings:
        name = metadata.get("section_id") or metadata.get("heading") or block.parent_section or "body"
        yield (str(name),), 0, len(block.text)
        return
    if headings[0].start():
        yield ("preamble",), 0, headings[0].start()
    path = []
    for index, heading in enumerate(headings):
        depth = len(heading[1])
        path = [(level, name) for level, name in path if level < depth]
        path.append((depth, heading[2].strip()))
        end = headings[index+1].start() if index+1 < len(headings) else len(block.text)
        yield tuple(name for _, name in path), heading.start(), end


class VersionIndex:
    def __init__(self, blocks, metadata, parts, versions):
        self.blocks, self.metadata = blocks, metadata
        self.parts, self.versions = parts, versions
        self.latest_orders = {}
        for (family, _), info in versions.items():
            if info["order"]:
                self.latest_orders[family] = max(self.latest_orders.get(family, ()), info["order"])

    @classmethod
    def from_blocks(cls, blocks, metadata=None, documents=None):
        blocks = tuple(blocks)
        metadata = metadata or {}
        documents = documents or DocumentIndex.from_blocks(blocks)
        parts, versions, by_key = {}, {}, {block.key: block for block in blocks}
        tables = {}
        for table in documents.tables.values():
            tables.setdefault(table.source.source_key, []).append(table)
        for block in blocks:
            info = metadata.get(block.key, {})
            family = info.get("document_family", block.key.document_id)
            key = family, block.key.version
            version = versions.setdefault(key, {"keys": [], "order": tuple(info.get("version_order", ())),
                "valid_from": info.get("valid_from"), "valid_to": info.get("valid_to"),
                "current": info.get("current")})
            version["keys"].append(block.key)
            grouped = parts.setdefault(key, {})
            sections = tuple(_sections(block, info))
            for path, start, end in sections:
                address = ("section", *path)
                grouped.setdefault(address, []).append(VersionPart(address, block.text[start:end],
                    (_span(block, start, end),)))
            for table in tables.get(block.key, ()):
                table_start = table.source.start - (block.span[0] if block.span else 0)
                path = next(path for path, start, end in sections if start <= table_start < end)
                table_name = table.caption if table.caption != block.title else ""
                for row in table.rows:
                    for index, cell in enumerate(row[1:], 1):
                        address = ("field", *path, table_name, row[0].text, table.headers[index])
                        spans = _unique_spans((row[0].source, cell.source,
                            *((cell.header,) if cell.header else ()), *cell.qualifiers))
                        grouped.setdefault(address, []).append(VersionPart(address, cell.text, spans))
        return cls(by_key, metadata, parts, versions)

    def legal_keys(self, *, families=(), versions=(), as_of=None, current=False):
        families, versions = set(families), set(versions)
        selected = []
        for (family, version), info in self.versions.items():
            if families and family not in families:
                continue
            if versions and version not in versions:
                continue
            if as_of is not None:
                if info["valid_from"] is not None and as_of < info["valid_from"]:
                    continue
                if info["valid_to"] is not None and as_of >= info["valid_to"]:
                    continue
            if current and not versions and as_of is None:
                if info["current"] is False:
                    continue
                if info["current"] is None:
                    latest = self.latest_orders.get(family, ())
                    if latest and info["order"] != latest:
                        continue
            selected.extend(info["keys"])
        return tuple(selected)

    def adjacent(self, family, version, direction):
        """Resolve a predecessor/successor on the source-declared version axis."""
        members = {key[1]: info for key, info in self.versions.items() if key[0] == family}
        axis = "order" if members[version]["order"] else "valid_from"
        anchor = members[version][axis]
        if anchor is None:
            return None
        values = [(info[axis], name) for name, info in members.items() if info[axis] is not None
                  and (info[axis] < anchor if direction == "before" else info[axis] > anchor)]
        return (max(values) if direction == "before" else min(values))[1] if values else None

    def changes(self, family, before_version, after_version):
        before = self.parts[family, before_version]
        after = self.parts[family, after_version]
        result = []
        for address in dict.fromkeys((*before, *after)):
            old, new = before.get(address, ()), after.get(address, ())
            old_text, new_text = "\n".join(part.text for part in old), "\n".join(part.text for part in new)
            if old_text == new_text:
                continue
            old_spans = tuple(span for part in old for span in part.spans)
            new_spans = tuple(span for part in new for span in part.spans)
            if address[0] == "section" and len(old) == len(new) == 1:
                old_spans, new_spans = self._changed_spans(old[0], new[0])
            result.append(VersionChange(family, before_version, after_version, address,
                old_spans, new_spans, "added" if not old else "removed" if not new else "modified"))
        return tuple(result)

    @staticmethod
    def _changed_spans(before, after):
        """Locate changed original lines while retaining the enclosing section relation."""
        left, right = before.text.splitlines(keepends=True), after.text.splitlines(keepends=True)
        left_offsets, right_offsets = [0], [0]
        for line in left:
            left_offsets.append(left_offsets[-1] + len(line))
        for line in right:
            right_offsets.append(right_offsets[-1] + len(line))
        from controller.context_schema import SourceSpan
        old, new = [], []
        for operation, i, j, k, m in SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
            if operation == "equal":
                continue
            for target, part, offsets, start, end in (
                    (old, before, left_offsets, i, j), (new, after, right_offsets, k, m)):
                if start < end:
                    source = part.spans[0]
                    first, last = source.start + offsets[start], source.start + offsets[end]
                    target.append(SourceSpan(f"{source.span_id}:change:{first}:{last}",
                                             source.source_key, first, last))
        return tuple(old), tuple(new)

    def paired_evidence(self, family, before_version, after_version, addresses=()):
        changes = self.changes(family, before_version, after_version)
        return tuple(change for change in changes if not addresses or change.address in addresses)

    def paired_units(self, family, before_version, after_version, selected_spans=()):
        """Retrieve both sides of matching sections/fields, including unchanged content."""
        from controller.context_schema import EvidenceUnit
        before, after = self.parts[family, before_version], self.parts[family, after_version]
        addresses = tuple(dict.fromkeys((*before, *after)))
        if selected_spans:
            addresses = tuple(address for address in addresses if any(
                source.source_key == selected.source_key and source.start < selected.end
                and selected.start < source.end
                for part in (*before.get(address, ()), *after.get(address, ()))
                for source in part.spans for selected in selected_spans))
        grouped = {}
        for address in addresses:
            left, right = before.get(address, ()), after.get(address, ())
            if selected_spans and (len(left) > 1 or len(right) > 1):
                pairs = _part_pairs(left, right)
                parts = tuple(part for pair in pairs if any(
                    source.source_key == selected.source_key and source.start < selected.end
                    and selected.start < source.end
                    for part in pair for source in part.spans for selected in selected_spans)
                    for part in pair)
            else:
                parts = (*left, *right)
            members = []
            for part in parts:
                for span in part.spans:
                    block = self.blocks[span.source_key]
                    base = block.span[0] if block.span else 0
                    identity = f"version:{span.span_id}"
                    grouped.setdefault(identity, [span, block.text[span.start-base:span.end-base], set()])
                    members.append(identity)
            for identity in members:
                grouped[identity][2].update(member for member in members if member != identity)
        return tuple(EvidenceUnit(identity, (span,), text, " / ".join((family, span.source_key.version)),
            dependencies=tuple(sorted(dependencies)), document_family=family)
            for identity, (span, text, dependencies) in grouped.items())


def _part_pairs(before, after):
    """Monotone content alignment for sections split across source passages."""
    m, n = len(before), len(after)
    left = [tuple(re.findall(r"\w+|[^\w\s]", part.text)) for part in before]
    right = [tuple(re.findall(r"\w+|[^\w\s]", part.text)) for part in after]
    scores = [[0.] * (n + 1) for _ in range(m + 1)]
    moves = {}
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            similarity = SequenceMatcher(None, left[i-1], right[j-1], autojunk=False).ratio()
            choices = ((scores[i-1][j], "before"), (scores[i][j-1], "after"),
                       (scores[i-1][j-1] + similarity, "pair"))
            scores[i][j], moves[i, j] = max(choices, key=lambda item: item[0])
    pairs, i, j = [], m, n
    while i or j:
        move = moves[i, j] if i and j else "before" if i else "after"
        if move == "pair":
            pairs.append((before[i-1], after[j-1]))
            i, j = i-1, j-1
        elif move == "before":
            pairs.append((before[i-1],))
            i -= 1
        else:
            pairs.append((after[j-1],))
            j -= 1
    return tuple(reversed(pairs))
