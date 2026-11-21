"""Version, section and field correspondence over original source addresses."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from adapters.document_compute import DocumentIndex, _span, _unique_spans


def catalogue_versions(catalogue, families=()):
    """Compact source-declared version axis, shared by runtime and training."""
    result = {}
    for row in sorted(catalogue, key=lambda row: (row["family"], tuple(row.get("version_order") or ()))):
        if not families or row["family"] in families:
            result.setdefault(row["family"], []).append(row["version"])
    return result


@dataclass(frozen=True)
class VersionPart:
    address: tuple[str, ...]
    text: str
    spans: tuple


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


def _content_shingles(text):
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 5:
        return len(words), set(words)
    return len(words), set(zip(words, words[1:], words[2:], words[3:], words[4:]))


class VersionIndex:
    def __init__(self, blocks, metadata, parts, versions, *, incremental_families=frozenset()):
        self.blocks, self.metadata = blocks, metadata
        self.parts, self.versions = parts, versions
        self.incremental_families = frozenset(incremental_families)
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
        # Distinguish cumulative snapshots (latest supersedes earlier) from incremental
        # release-note series (consecutive versions share little content and lack valid_to).
        incremental = set()
        by_family = {}
        for (family, version), info in versions.items():
            by_family.setdefault(family, []).append((info["order"], info["valid_from"] or "", version, info))
        for family, items in by_family.items():
            if len(items) < 2 or any(info["valid_to"] is not None for _, _, _, info in items):
                continue
            items.sort()
            shingles = [_content_shingles("\n".join(by_key[k].text for k in info["keys"]))
                        for _, _, _, info in items]
            word_counts = sorted(count for count, _ in shingles)
            if word_counts[len(word_counts) // 2] < 100:
                continue
            retentions = sorted(len(left & right) / max(1, len(left))
                                for (_, left), (_, right) in zip(shingles, shingles[1:]))
            if retentions[len(retentions) // 2] < 0.5:
                incremental.add(family)
        return cls(by_key, metadata, parts, versions, incremental_families=incremental)

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
            if current and not versions and as_of is None and family not in self.incremental_families:
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

    def _units_from_part_groups(self, part_groups):
        from controller.schema import EvidenceUnit
        grouped = {}
        for parts in part_groups:
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
        units = []
        for identity, (span, text, dependencies) in grouped.items():
            family = self.metadata.get(span.source_key, {}).get("document_family", span.source_key.document_id)
            units.append(EvidenceUnit(identity, (span,), text, " / ".join((family, span.source_key.version)),
                dependencies=tuple(sorted(dependencies)), document_family=family))
        return tuple(units)

    def paired_units(self, family, before_version, after_version, selected_spans=()):
        """Retrieve both sides of matching sections/fields, including unchanged content."""
        before, after = self.parts[family, before_version], self.parts[family, after_version]
        addresses = tuple(dict.fromkeys((*before, *after)))
        pairs = []
        for address in addresses:
            left, right = before.get(address, ()), after.get(address, ())
            if address[0] == "section" and (not left or not right):
                continue
            if len(left) > 1 or len(right) > 1:
                pairs.extend(_part_pairs(left, right))
            else:
                pairs.append((*left, *right))
        # A changed heading is part of the version change, not a new document identity.
        unmatched_before = tuple(part for address, parts in before.items()
                                 if address[0] == "section" and address not in after for part in parts)
        unmatched_after = tuple(part for address, parts in after.items()
                                if address[0] == "section" and address not in before for part in parts)
        pairs.extend(_part_pairs(unmatched_before, unmatched_after))
        matched = [parts for parts in pairs if not selected_spans or any(
            source.source_key == selected.source_key and source.start < selected.end
            and selected.start < source.end
            for part in parts for source in part.spans for selected in selected_spans)]
        return self._units_from_part_groups(matched)

    def change_boundary_units(self, selected_spans, *, families=(), limit=6):
        """Locate earliest introduction and changed consecutive versions for retrieved sections."""
        if not selected_spans:
            return ()
        allowed = set(families)
        ordered_by_family = {}
        for (family, version), info in self.versions.items():
            if allowed and family not in allowed:
                continue
            ordered_by_family.setdefault(family, []).append((info["order"], info["valid_from"] or "", version))
        for items in ordered_by_family.values():
            items.sort()
        touched = []
        seen_addr = set()
        for span in selected_spans:
            info = self.metadata.get(span.source_key, {})
            family = info.get("document_family", span.source_key.document_id)
            if family not in ordered_by_family or len(ordered_by_family[family]) < 2:
                continue
            for address, parts in self.parts.get((family, span.source_key.version), {}).items():
                if address[0] != "section" or address[-1] in {"body", "preamble"}:
                    continue
                if (family, address) in seen_addr:
                    continue
                if any(s.source_key == span.source_key and s.start < span.end and span.start < s.end
                       for part in parts for s in part.spans):
                    seen_addr.add((family, address))
                    touched.append((family, address))
                    if len(touched) >= limit:
                        break
            if len(touched) >= limit:
                break
        chosen_groups = []
        for family, address in touched:
            ordered = [v for _, _, v in ordered_by_family[family]]
            present = [(idx, v, self.parts[family, v][address])
                       for idx, v in enumerate(ordered) if address in self.parts.get((family, v), {})]
            if not present:
                continue
            section_groups = [present[0][2]]
            for (_, _, left_parts), (_, _, right_parts) in zip(present, present[1:]):
                left_text = "\n".join(p.text.strip() for p in left_parts)
                right_text = "\n".join(p.text.strip() for p in right_parts)
                if left_text != right_text:
                    section_groups.append((*left_parts, *right_parts))
                    if len(section_groups) >= 3:
                        break
            chosen_groups.extend(section_groups)
        return self._units_from_part_groups(chosen_groups)


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
