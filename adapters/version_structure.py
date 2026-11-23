"""Version, section and field correspondence over original source addresses."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from adapters.document_compute import DocumentIndex, _span, _unique_spans


_IDENTIFIER = re.compile(
    r"(?<!\w)(?:[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+|[A-Z]{2,}-\d+|"
    r"[A-Z][A-Z\d]*(?:_[A-Z\d]+)+|[A-Za-z][a-z]+(?:[A-Z][A-Za-z\d]*)+|"
    r"[a-z][\w-]*\.(?:js|ts|py|java|scala|go|rs|sh|json|yaml|yml|xml|md))(?!\w)"
)


def technical_identifiers(text: str) -> tuple[str, ...]:
    """Extract literal code, config, file, or issue identifiers, excluding product names like Node.js."""
    return tuple(
        symbol for symbol in dict.fromkeys(_IDENTIFIER.findall(text))
        if not re.fullmatch(r"[A-Z][a-z]+\.js", symbol)
    )


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


class VersionIndex:
    def __init__(self, blocks, metadata, parts, versions):
        self.blocks, self.metadata = blocks, metadata
        self.parts, self.versions = parts, versions
        self.latest_orders = {}
        families = {}
        for (family, version), info in versions.items():
            families.setdefault(family, []).append(version)
            if info["order"]:
                self.latest_orders[family] = max(self.latest_orders.get(family, ()), info["order"])
        self.editions = {
            family: tuple(sorted(
                members,
                key=lambda ver: (versions[family, ver]["order"], versions[family, ver]["valid_from"] or "", ver),
            ))
            for family, members in families.items()
        }
        self.cumulative_families = self._detect_cumulative_families()

    def _detect_cumulative_families(self):
        cumulative = set()
        for family, ordered in self.editions.items():
            if len(ordered) <= 1:
                cumulative.add(family)
                continue
            overlaps = []
            for v1, v2 in zip(ordered, ordered[1:]):
                s1 = {a for a in self.parts.get((family, v1), ())
                      if a[0] == "section" and a not in {("section", "body"), ("section", "preamble")}}
                s2 = {a for a in self.parts.get((family, v2), ())
                      if a[0] == "section" and a not in {("section", "body"), ("section", "preamble")}}
                if s1 or s2:
                    overlaps.append(len(s1 & s2) / max(1, len(s1 | s2)))
                else:
                    t1 = {w.lower() for k in self.versions[family, v1]["keys"]
                          for w in re.findall(r"[A-Za-z_]\w{3,}", self.blocks[k].text)}
                    t2 = {w.lower() for k in self.versions[family, v2]["keys"]
                          for w in re.findall(r"[A-Za-z_]\w{3,}", self.blocks[k].text)}
                    overlaps.append(len(t1 & t2) / max(1, len(t1 | t2)))
            if sum(overlaps) / len(overlaps) >= 0.50:
                cumulative.add(family)
        return frozenset(cumulative)

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
            if current and not versions and as_of is None and family in self.cumulative_families:
                if info["current"] is False:
                    continue
                if info["current"] is None:
                    latest = self.latest_orders.get(family, ())
                    if latest and info["order"] != latest:
                        continue
            selected.extend(info["keys"])
        return tuple(selected)

    def _specific_symbols(self, question, family, editions):
        """Keep technical identifiers that discriminate sections rather than naming the whole product."""
        symbols = technical_identifiers(question)
        fam_tokens = set(re.findall(r"[a-z\d]+", family.lower()))
        specific = []
        for symbol in symbols:
            sym_parts = [p for p in re.findall(r"[a-z\d]+", symbol.lower()) if p not in {"js", "py", "md", "pdf"}]
            if sym_parts and all(p in fam_tokens or (p + "js") in fam_tokens for p in sym_parts):
                continue
            literal = re.compile(r"(?<!\w)" + re.escape(symbol) + r"(?!\w)", re.IGNORECASE)
            editions_hit = 0
            max_passage_share = 0.0
            for ver in editions:
                keys = self.versions[family, ver]["keys"]
                hits = sum(1 for k in keys if literal.search(self.blocks[k].text))
                if hits:
                    editions_hit += 1
                if len(keys) >= 4:
                    max_passage_share = max(max_passage_share, hits / len(keys))
            if editions_hit and max_passage_share <= 0.35:
                idf = 1.0 + len(editions) / editions_hit
                specific.append((symbol, literal, idf))
        return tuple(specific)

    def symbol_version_presence(self, question, families=()):
        """Report ordered supplied editions containing or preceding queried technical identifiers."""
        target_families = set(families) if families else set(self.editions)
        presence = {}
        for family in sorted(target_families):
            editions = self.editions.get(family, ())
            if len(editions) <= 1:
                continue
            for symbol, literal, _ in self._specific_symbols(question, family, editions):
                present, absent_before = [], []
                seen_present = False
                for ver in editions:
                    has = any(literal.search(self.blocks[k].text) for k in self.versions[family, ver]["keys"])
                    if has:
                        present.append(ver)
                        seen_present = True
                    elif not seen_present:
                        absent_before.append(ver)
                if present and absent_before:
                    presence[symbol] = {
                        "family": family,
                        "earliest_supplied_version": present[0],
                        "present_versions": present,
                        "absent_earlier_versions": absent_before,
                    }
        return presence

    def evolution_units(self, question, candidate_spans, families=(), max_pairs=3):
        """Locate version boundary pairs where queried symbols or retrieved sections appear or change."""
        from controller.schema import EvidenceUnit
        target_families = set(families) if families else set(self.cumulative_families)
        if not target_families:
            return ()
        matched_addresses = {}
        for family in sorted(target_families):
            editions = self.editions.get(family, ())
            if len(editions) <= 1:
                continue
            patterns = self._specific_symbols(question, family, editions)
            scored_addrs = {}
            for ver in editions:
                for addr, part_list in self.parts.get((family, ver), {}).items():
                    if addr[0] != "section" or addr in {("section", "body"), ("section", "preamble")}:
                        continue
                    for part in part_list:
                        if len(part.text) > 4000:
                            continue
                        sym_score = sum(idf for _, pat, idf in patterns if pat.search(part.text))
                        span_hits = sum(
                            1 for src in part.spans for cand in candidate_spans[:12]
                            if src.source_key == cand.source_key
                            and src.start < cand.end and cand.start < src.end
                        )
                        if sym_score > 0 or span_hits > 0:
                            scored_addrs[addr] = max(scored_addrs.get(addr, (0.0, 0)), (sym_score, span_hits))
            if scored_addrs:
                top_addrs = [a for a, _ in sorted(scored_addrs.items(), key=lambda x: x[1], reverse=True)[:2]]
                matched_addresses[family] = (editions, top_addrs)

        grouped = {}
        pairs_added = 0
        for family, (editions, top_addrs) in matched_addresses.items():
            for addr in top_addrs:
                prev_text = None
                prev_parts = ()
                for ver in editions:
                    cur_parts = tuple(p for p in self.parts.get((family, ver), {}).get(addr, ()) if len(p.text) <= 4000)
                    cur_text = "\n".join(re.sub(r"\s+", " ", p.text.strip()) for p in cur_parts) if cur_parts else None
                    if cur_text is not None and cur_text != prev_text:
                        chosen_parts = (*prev_parts[:1], *cur_parts[:1])
                        members = []
                        for part in chosen_parts:
                            for span in part.spans:
                                block = self.blocks[span.source_key]
                                base = block.span[0] if block.span else 0
                                identity = f"version:{span.span_id}"
                                grouped.setdefault(identity, [span, block.text[span.start - base:span.end - base], set()])
                                members.append(identity)
                        for identity in members:
                            grouped[identity][2].update(m for m in members if m != identity)
                        pairs_added += 1
                        if pairs_added >= max_pairs:
                            break
                    if cur_text is not None:
                        prev_text, prev_parts = cur_text, cur_parts
                if pairs_added >= max_pairs:
                    break
        return tuple(
            EvidenceUnit(
                identity, (span,), text, " / ".join((family, span.source_key.version)),
                dependencies=tuple(sorted(deps)), document_family=family,
            )
            for identity, (span, text, deps) in grouped.items()
        )

    def missing_identifiers(self, question, version_keys, families):
        """Find named symbols documented in the family but absent from the requested editions."""
        missing = []
        for symbol in technical_identifiers(question):
            literal = re.compile(r"(?<!\w)" + re.escape(symbol) + r"(?!\w)", re.IGNORECASE)
            if any(literal.search(self.blocks[key].text) for key in version_keys):
                continue
            if any(literal.search(self.blocks[key].text)
                   for (family, _), info in self.versions.items() if family in families
                   for key in info["keys"]):
                missing.append(symbol)
        return tuple(missing)

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

    def paired_units(self, family, before_version, after_version, selected_spans=()):
        """Retrieve both sides of matching sections/fields, including unchanged content."""
        from controller.schema import EvidenceUnit
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
        grouped = {}
        for parts in pairs:
            if selected_spans and not any(
                source.source_key == selected.source_key and source.start < selected.end
                and selected.start < source.end
                for part in parts for source in part.spans for selected in selected_spans):
                continue
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
