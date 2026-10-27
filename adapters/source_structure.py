"""Source list hierarchy anchored to literal character spans, without rewriting.

The ingestion adapter supplies author HTML/text IDs. Reading serialization adds
only list tags; retrieval continues to use the original text. This follows the
structure-aware serialization principle used by Docling, without its conversion
pipeline when the source already supplies HTML and exact text offsets.
"""
from __future__ import annotations

from html import escape
from html.parser import HTMLParser


class _AuthorLists(HTMLParser):
    def __init__(self, text, spans):
        super().__init__()
        self.text, self.spans = text, spans
        self.roots, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"ul", "ol", "li"}:
            node = {"tag": tag, "children": [], "ranges": []}
            names = {"start", "type", "reversed"} if tag == "ol" else {"value"} if tag == "li" else set()
            if kept := {name: value for name, value in attrs.items() if name in names}:
                node["attrs"] = kept
            (self.stack[-1]["children"] if self.stack else self.roots).append(node)
            self.stack.append(node)
        if self.stack and "text_id" in attrs:
            span = self.spans[attrs["text_id"]]
            start, end = span["start_sp"], span["end_sp"]
            if not 0 <= start < end <= len(self.text) or self.text[start:end] != span["text_sp"]:
                raise ValueError("author HTML span does not match the original text")
            for node in self.stack:
                node["ranges"].append((start, end))

    def handle_endtag(self, tag):
        if tag in {"ul", "ol", "li"}:
            if not self.stack or self.stack[-1]["tag"] != tag:
                raise ValueError("unbalanced author list markup")
            self.stack.pop()


def author_lists(text, html, spans):
    """Read list nesting from annotated HTML, never dialogue grounding or labels."""
    parser = _AuthorLists(text, spans)
    parser.feed(html)
    parser.close()
    if parser.stack:
        raise ValueError("unclosed author list markup")

    def anchored(nodes):
        result = []
        for node in nodes:
            ranges = node.pop("ranges")
            if not ranges:  # Empty/unannotated markup adds no source information.
                continue
            node.update(start=min(start for start, _ in ranges), end=max(end for _, end in ranges),
                        children=anchored(node["children"]))
            result.append(node)
        return result

    return anchored(parser.roots)


def section_lists(nodes, start, end):
    """Retain complete lists within a section; keep original section boundaries.

    A list crossing sections is not presented as a complete list. Complete nested
    lists can still be retained. All source words remain in the literal section.
    """
    def shift(item):
        return {**item, "start": item["start"] - start, "end": item["end"] - start,
                "children": [shift(child) for child in item["children"]]}

    result = []
    for node in nodes:
        if start <= node["start"] < node["end"] <= end and node["tag"] != "li":
            result.append(shift(node))
        elif node["start"] < end and start < node["end"]:
            result.extend(section_lists(node["children"], start, end))
    return result


def render_source_lists(text, nodes):
    """Insert source layout once at corpus load; preserve every original character.

    The returned body is the evidence identity used by both APPEND and REBUILD,
    so source presentation changes cannot alias an old cached physical body.
    """
    def render(start, end, children):
        result, position = [], start
        for node in children:
            tag, first, last = node["tag"], node["start"], node["end"]
            # Author HTML sometimes nests lists without an intervening li.
            # Preserve that source hierarchy instead of imposing a new grammar.
            if tag not in {"ul", "ol", "li"} or not position <= first < last <= end:
                raise ValueError("source lists must be nested, ordered original-text spans")
            names = {"start", "type", "reversed"} if tag == "ol" else {"value"} if tag == "li" else set()
            attrs = node.get("attrs", {})
            if not attrs.keys() <= names:
                raise ValueError("unsupported source list attribute")
            attributes = ''.join(f' {name}' if value is None else f' {name}="{escape(str(value), quote=True)}"'
                                 for name, value in sorted(attrs.items()))
            result.extend((text[position:first], f"\n<{tag}{attributes}>\n",
                           render(first, last, node["children"]), f"\n</{tag}>\n"))
            position = last
        result.append(text[position:end])
        return ''.join(result)

    return render(0, len(text), nodes)
