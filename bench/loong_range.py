"""Export a preselected Loong cohort from complete official ZIP members via Range."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
from threading import Lock
import time
import uuid
import zipfile
import zlib

import httpx
import yaml

from adapters.loong import COMMIT, DOCUMENTS_URL, PROTOCOL, Documents, SourceResolutionError, task
from training.reader_data import digest, file_digest, rows, write_rows


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def select(args):
    original = rows(args.questions)
    population = [row for row in original if row["set"] == args.length_set]
    selected = sorted(population, key=lambda row: digest(
        [args.seed, "loong", "test", row["id"]]))[:args.count]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output/"selected_questions.jsonl", selected)
    manifest = {"status": "selected_before_document_fetch", "seed": args.seed,
        "selection": "sort sha256(canonical_json([seed,'loong','test',original_id])); first count",
        "selection_uses": ["original_id", "length_set"], "uses_answers_for_selection": False,
        "questions": str(Path(args.questions).resolve()), "questions_sha256": file_digest(args.questions),
        "upstream_commit": COMMIT, "length_set": args.length_set, "population": len(population),
        "requested": args.count, "selected_ids": [row["id"] for row in selected],
        "sample_not_full_length_group": len(selected) < len(population),
        "selection_matches": "bench.a_final_batch.prepare seed/name=loong/split=test/conversation_id"}
    write(output/"cohort_plan.json", manifest)
    return selected, manifest


class HTTPRangeFile(io.RawIOBase):
    """A seekable ZIP byte source; every network read must be an exact HTTP 206."""

    def __init__(self, url, directory, *, timeout=180, workers=8, block_bytes=262144):
        super().__init__()
        self.url, self.position = url, 0
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex
        self.block_bytes, self.workers = block_bytes, workers
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.receipt_lock = Lock()
        self.client = httpx.Client(follow_redirects=True, timeout=timeout,
                                   headers={"Accept-Encoding": "identity"})
        self.receipts, self.cache = [], {}
        started = time.perf_counter()
        response = self.client.head(url)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.etag = response.headers["ETag"]
        self.identity = {"url": url, "final_url": str(response.url), "etag": self.etag,
            "archive_bytes": self.size, "last_modified": response.headers.get("Last-Modified"),
            "accept_ranges": response.headers.get("Accept-Ranges"),
            "head_elapsed_ms": (time.perf_counter()-started)*1000,
            "head_status": response.status_code, "head_headers": dict(response.headers),
            "range_workers": workers, "range_block_bytes": block_bytes,
            "run_id": self.run_id, "complete_zip_downloaded": False}
        write(self.directory/(self.run_id+".identity.json"), self.identity)
        # A real range response establishes support; HEAD's advisory flag is insufficient.
        self._range(0, 1)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = (offset if whence == io.SEEK_SET else self.position+offset
                         if whence == io.SEEK_CUR else self.size+offset)
        if self.position < 0:
            raise ValueError("negative ZIP byte offset")
        return self.position

    def _range(self, start, stop):
        key = (start, stop)
        if key in self.cache:
            return self.cache[key]
        if stop-start > self.block_bytes:
            chunks = [(offset, min(stop, offset+self.block_bytes))
                      for offset in range(start, stop, self.block_bytes)]
            raw = b"".join(self.executor.map(lambda bounds: self._fetch_range(*bounds), chunks))
            self.cache[key] = raw
            return raw
        return self._fetch_range(start, stop)

    def _fetch_range(self, start, stop):
        key = (start, stop)
        if key in self.cache:
            return self.cache[key]
        requested = f"bytes={start}-{stop-1}"
        started = time.perf_counter()
        expected = f"bytes {start}-{stop-1}/{self.size}"
        with self.client.stream("GET", self.url,
                                headers={"Range": requested, "If-Match": self.etag}) as response:
            matching = (response.status_code == 206 and response.headers.get("Content-Range") == expected
                        and response.headers.get("ETag") == self.etag)
            raw = response.read() if matching else b""
        from hashlib import sha256
        raw_sha256 = sha256(raw).hexdigest()
        path = self.directory/f"{self.run_id}.{start}-{stop-1}.{raw_sha256}.bin"
        path.write_bytes(raw)
        record = {"run_id": self.run_id, "range": requested, "status": response.status_code,
            "content_range": response.headers.get("Content-Range"), "etag": response.headers.get("ETag"),
            "bytes": len(raw), "sha256": raw_sha256, "raw_bytes": str(path.resolve()),
            "elapsed_ms": (time.perf_counter()-started)*1000}
        with self.receipt_lock:
            self.receipts.append(record)
            with (self.directory/"requests.jsonl").open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False)+"\n")
        response.raise_for_status()
        if (response.status_code != 206 or response.headers.get("Content-Range") != expected
                or len(raw) != stop-start or response.headers.get("ETag") != self.etag):
            raise ValueError(f"official ZIP Range identity mismatch: {record}")
        self.cache[key] = raw
        return raw

    def read(self, size=-1):
        stop = self.size if size < 0 else min(self.size, self.position+size)
        if self.position >= stop:
            return b""
        data = self._range(self.position, stop)
        self.position = stop
        return data

    def close(self):
        self.executor.shutdown(wait=True)
        self.client.close()
        self.cache.clear()
        super().close()


class RangeDocuments(Documents):
    """Reuse the pinned adapter's source resolution and complete text assembly."""

    def __init__(self, url, directory, *, timeout=180, workers=8, block_bytes=262144):
        self.source_uri, self.directory = url, Path(directory)
        self.stream = HTTPRangeFile(url, self.directory/"ranges", timeout=timeout,
                                    workers=workers, block_bytes=block_bytes)
        self.archive = zipfile.ZipFile(self.stream, metadata_encoding="utf-8")
        self.names = tuple(info.filename for info in self.archive.infolist()
            if not info.is_dir() and info.filename.startswith("doc/")
            and not info.filename.endswith(".DS_Store"))
        if len(set(self.names)) != len(self.names):
            raise ValueError("duplicate document members in archive")
        self.legal, self._texts, self.members = None, {}, {}
        write(self.directory/"central_directory.json", [{"filename": info.filename,
            "CRC": info.CRC, "compress_type": info.compress_type, "compressed_bytes": info.compress_size,
            "original_bytes": info.file_size, "local_header_offset": info.header_offset,
            "flag_bits": info.flag_bits} for info in self.archive.infolist()])

    def read(self, name):
        if name not in self._texts:
            from hashlib import sha256
            info = self.archive.getinfo(name)
            raw = self.archive.read(name)  # ZipFile verifies the published member CRC.
            path = self.directory/"members"/(sha256(name.encode()).hexdigest()+".bin")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            record = {"member": name, "original_bytes": len(raw), "compressed_bytes": info.compress_size,
                "header_offset": info.header_offset, "compress_type": info.compress_type,
                "published_crc32": f"{info.CRC:08x}", "observed_crc32": f"{zlib.crc32(raw):08x}",
                "crc_verified": True, "sha256": sha256(raw).hexdigest(), "raw_bytes": str(path.resolve()),
                "archive_etag": self.stream.etag, "complete_member": True}
            self.members[name] = record
            with (self.directory/"members.jsonl").open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False)+"\n")
            self._texts[name] = raw.decode("utf-8")
        return self._texts[name]

    def close(self):
        super().close()
        self.stream.close()


def export(args):
    selected, plan = select(args)
    output = Path(args.output)
    exported, issues = [], []
    started = time.perf_counter()
    with RangeDocuments(args.documents_url, output/"source", timeout=args.timeout,
                        workers=args.workers, block_bytes=args.block_bytes) as documents:
        with (output/"tasks.jsonl").open("w") as stream:
            for row in selected:
                try:
                    value = task(row, documents, revision=COMMIT+":"+PROTOCOL)
                except SourceResolutionError as error:
                    issues.append({"task_id": row["id"], "document": error.document,
                                   "candidates": error.candidates, "error": str(error)})
                    continue
                stream.write(json.dumps(value, ensure_ascii=False)+"\n")
                stream.flush()
                exported.append(value)
        requests = documents.stream.receipts
        manifest = {**plan, "status": "partial_source_resolution" if issues else "completed",
            "protocol": PROTOCOL, "documents": documents.stream.identity,
            "range_requests": len(requests), "downloaded_bytes": sum(row["bytes"] for row in requests),
            "range_receipts": str((output/"source/ranges/requests.jsonl").resolve()),
            "central_directory_sha256": file_digest(output/"source/central_directory.json"),
            "member_receipts": list(documents.members.values()),
            "exported": len(exported), "exported_ids": [row["task_id"] for row in exported],
            "source_issues": issues, "tasks_sha256": file_digest(output/"tasks.jsonl"),
            "domains": dict(Counter(row["Collection"] for row in exported)),
            "source_instances": sum(len(row["contexts"]) for row in exported),
            "all_allowed_documents_complete": not issues,
            "text_assembly": "adapters.loong.task/Documents.context; no document or source text removed",
            "model_calls": 0, "elapsed_ms": (time.perf_counter()-started)*1000}
    write(output/"cohort_manifest.json", manifest)
    write(output/"manifest.json", manifest)
    write(output/"source_issues.json", issues)
    write(output/"split_manifest.json", {"dev": [], "test": manifest["exported_ids"],
        "scope": "deterministic sample of public set1; not all 323 and not a complete ZIP download"})
    spec = {"inputs": {"dev": args.dev_input, "test": str((output/"tasks.jsonl").resolve())},
        "split_kind": "independent_training_sources_dev_and_preselected_public_set1_sample_test",
        "coverage_mode": "all_supplied_documents", "primary_metric": "loong_reference_score",
        "normalization": 100.0, "weight": .25,
        "source_cohort_manifest": str((output/"cohort_manifest.json").resolve())}
    (output/"study_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "export"))
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--length-set", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--documents-url", default=DOCUMENTS_URL)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 9))
    parser.add_argument("--block-bytes", type=int, default=262144)
    parser.add_argument("--dev-input", default="runs/study/evaluation/long_dev/tasks.jsonl")
    args = parser.parse_args(argv)
    result = select(args)[1] if args.command == "select" else export(args)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 2 if result.get("source_issues") else 0


if __name__ == "__main__":
    raise SystemExit(main())
