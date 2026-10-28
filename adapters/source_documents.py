"""Recover literal paragraph context from the matching original source document.

This reader supplies context to existing passages, without changing retrieval IDs,
reindexing, generating headings or rewriting source text. Only whitespace may vary
between an ingested passage and its original document during alignment.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path
import re
import zipfile


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    text: str
    source: str
    title: str
    paragraphs: tuple[tuple[int, int], ...]
    corpus_id: str = ""
    snapshot_id: str = ""
    version: str = ""

    @cached_property
    def identity_sha256(self):
        value = (self.corpus_id, self.snapshot_id, self.version, self.document_id,
                 self.source, self.title, self.text)
        return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode('utf-8')).hexdigest()


def paragraph_spans(text):
    start = 0
    spans = []
    for separator in re.finditer(r"\n[ \t\r\n]*\n", text):
        if text[start:separator.start()].strip():
            spans.append((start, separator.start()))
        start = separator.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return tuple(spans)


class SourceDocuments:
    def __init__(self, records):
        self.documents = {}
        for row in records:
            row = {**row.get('metadata', {}), **row}
            document_id = str(row.get('document_id', row.get('_id', '')))
            if not document_id or document_id in self.documents:
                raise ValueError('original documents need distinct document_id values')
            text = row['text']
            self.documents[document_id] = SourceDocument(document_id, text,
                str(row.get('source_uri') or row.get('url') or row.get('source') or ''),
                str(row.get('title') or ''), paragraph_spans(text),
                str(row.get('corpus_id') or ''), str(row.get('snapshot_id') or ''),
                str(row.get('version') or ''))

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                members = [name for name in archive.namelist()
                           if name.endswith('.jsonl') and not name.startswith('__MACOSX/')]
                if len(members) != 1:
                    raise ValueError('source document archive must contain one JSONL corpus')
                with archive.open(members[0]) as stream:
                    return cls(json.loads(line) for line in stream if line.strip())
        with path.open(encoding='utf-8') as stream:
            return cls(json.loads(line) for line in stream if line.strip())

    def window(self, document_id, passage, *, corpus_id=None, snapshot_id=None,
               version=None, source=None):
        document = self.documents.get(document_id)
        if document is None or not passage.strip():
            return None, 'document_unavailable'
        # A sentence shared by revisions is alignment, not revision identity.
        # Corpus readers must bind the original to their declared snapshot. A
        # product version distinct from that snapshot additionally needs its own
        # exact version metadata; absent metadata never means "current/latest".
        if corpus_id is not None and document.corpus_id and document.corpus_id != corpus_id:
            return None, 'corpus_mismatch'
        if corpus_id is not None and not source and not document.corpus_id:
            return None, 'corpus_unconfirmed'
        if snapshot_id is not None:
            if not document.snapshot_id:
                return None, 'snapshot_unconfirmed'
            if document.snapshot_id != snapshot_id:
                return None, 'snapshot_mismatch'
        if version is not None:
            if not document.version and version != document.snapshot_id:
                return None, 'version_unconfirmed'
            declared_version = document.version or document.snapshot_id
            if not declared_version:
                return None, 'version_unconfirmed'
            if declared_version != version:
                return None, 'version_mismatch'
        if source:
            if not document.source:
                return None, 'source_unconfirmed'
            if document.source != source:
                return None, 'source_mismatch'
        # Whitespace is normalized only for locating a source span. Both the
        # original passage and the returned document substring remain untouched.
        pattern = re.compile(r'\s+'.join(re.escape(word) for word in passage.split()))
        match = pattern.search(document.text)
        if match is None:
            return None, 'passage_not_aligned'
        if pattern.search(document.text, match.start() + 1) is not None:
            return None, 'ambiguous_alignment'
        spans = document.paragraphs
        first = bisect_right([end for _, end in spans], match.start())
        last = bisect_left([start for start, _ in spans], match.end()) - 1
        # Complete the boundary paragraphs and retain two neighboring paragraphs:
        # e.g. an introductory list item and its separated command/condition.
        lo = spans[max(0, first - 2)][0]
        hi = spans[min(len(spans) - 1, last + 2)][1]
        if document.text[lo:hi].split() == passage.split():
            return None, 'already_complete'
        return (document, lo, hi), 'aligned'
