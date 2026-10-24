#!/usr/bin/env python3
"""Fetch the pinned SGLang source, verify hashes, apply and syntax-check the patch.

--source-only downloads the audited files for CPU patch verification. It does NOT
create an installable engine. Omit it to prepare a complete git checkout on Linux.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "engine_patches/source_manifest.json").read_text())
PATCH = ROOT / "engine_patches/0001-contextkv-protocol.patch"


def run(args, cwd, *, check=True):
    return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)


def prepare(checkout: Path, source_only: bool):
    checkout.mkdir(parents=True, exist_ok=True)
    commit = MANIFEST["commit"]
    if not (checkout / ".git").exists():
        if any(checkout.iterdir()):
            raise RuntimeError("Refusing to initialize a nonempty checkout directory")
        run(["git", "init", "-q"], checkout)
        if source_only:
            (checkout / ".contextkv-source-only").write_text(commit + "\n")
        else:
            run(["git", "remote", "add", "origin", MANIFEST["repository"] + ".git"], checkout)
            run(["git", "fetch", "--depth", "1", "origin", commit], checkout)
            run(["git", "checkout", "--detach", "FETCH_HEAD"], checkout)
    elif source_only != (checkout / ".contextkv-source-only").exists():
        raise RuntimeError("Source-only validation and full checkout destinations must differ")
    if not source_only:
        head = run(["git", "rev-parse", "HEAD"], checkout).stdout.strip()
        if head != commit:
            raise RuntimeError(f"Wrong SGLang commit: {head}; expected {commit}")
    reverse = run(["git", "apply", "--reverse", "--check", str(PATCH)], checkout, check=False)
    already_applied = reverse.returncode == 0
    if source_only and not already_applied:
        def fetch(item):
            rel, digest = item
            dest = checkout / rel
            if dest.exists():
                raw = dest.read_bytes()
            else:
                url = f"https://raw.githubusercontent.com/sgl-project/sglang/{commit}/{rel}"
                raw = urllib.request.urlopen(url, timeout=60).read()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise RuntimeError(f"Source checksum mismatch: {rel}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch, MANIFEST["original_sha256"].items()))
    if not already_applied:
        for rel, digest in MANIFEST["original_sha256"].items():
            if hashlib.sha256((checkout / rel).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Source checksum mismatch: {rel}")
        run(["git", "apply", "--check", str(PATCH)], checkout)
        run(["git", "apply", str(PATCH)], checkout)
    for rel in MANIFEST["changed_files"]:
        compile((checkout / rel).read_text(), str(checkout / rel), "exec")
    scheduler = (checkout / "python/sglang/srt/managers/scheduler.py").read_text()
    guard = scheduler.index("get_runtime(self).admit(recv_req, req)")
    ensure = scheduler.index("self.tree_cache.ensure_session_generation(", guard)
    assert guard < ensure
    runtime = (checkout / "python/sglang/srt/managers/contextkv_runtime.py").read_bytes()
    assert runtime == (ROOT / "engine_patches/contextkv_runtime.py").read_bytes()
    scope = (checkout / "python/sglang/srt/managers/contextkv_model_scope.py").read_bytes()
    assert scope == (ROOT / "engine_patches/contextkv_model_scope.py").read_bytes()
    # Check the exact pinned native observation/transport seams without importing CUDA.
    processor = (checkout / "python/sglang/srt/managers/scheduler_components/batch_result_processor.py").read_text()
    assert "result.copy_done.synchronize()" in processor
    assert "req.time_stats.set_prefill_finished_time()" in processor
    assert "req.time_stats.set_completion_time()" in processor
    cache = (checkout / "python/sglang/srt/mem_cache/unified_radix_cache.py").read_text()
    assert 'runtime.record_transfer("D2H", ack)' in cache
    assert 'runtime.record_transfer("H2D", ack)' in cache
    http = (checkout / "python/sglang/srt/entrypoints/http_server.py").read_text()
    publish = http.index('"publish_result", {"attempt_id": obj.contextkv_attempt_id, "response": ret}')
    assert publish < http.index("return orjson_response(ret)", publish)
    token_frame = http.index("dumps_json(normalized.tokens_before_receipt(out))")
    stream_publish = http.index('"publish_result", {"attempt_id": obj.contextkv_attempt_id, "response": out}')
    terminal_frame = http.index('yield b"data: " + dumps_json(out)', stream_publish)
    assert token_frame < stream_publish < terminal_frame
    assert 'if key not in {"contextkv_receipt", "contextkv_timing"}' in (checkout / "python/sglang/srt/managers/tokenizer_manager.py").read_text()
    common = (checkout / "python/sglang/srt/mem_cache/common.py").read_text()
    match_start = common.index("def match_kv_cache(")
    match_end = common.index("    return match_result", match_start)
    observer = common.index("runtime.record_native_match(req, len(token_ids))", match_start)
    assert match_start < common.index("req.num_matched_prefix_tokens =", match_start) < observer < match_end
    assert b"PrefillAdder._get_running_request_total_token_offset(offset_owner, req)" in runtime
    assert b"budget_requests = self.scheduler.running_batch.reqs" in runtime
    assert b"request_id=req.rid" in runtime
    backup_start = cache.index("    def _execute_and_commit_kv_backup(")
    backup_end = cache.index("    @staticmethod", backup_start)
    backup = cache[backup_start:backup_end]
    assert backup.index("runtime.allow_host_backup") < backup.index("self._execute_kv_backup(")
    assert (backup.index("self.tree_core.commit_backup(")
            < backup.index("self._track_write_through_node(")
            < backup.index("runtime.record_host_backup_admitted("))
    assert "contextkv_request=req" in cache
    assert "runtime.record_host_capacity_eviction(evicted)" in cache
    policy = (checkout / "python/sglang/srt/managers/schedule_policy.py").read_text()
    select = policy[policy.index("    def _select_prefill_admission("):
                    policy.index("    def _commit_prefill_admission(")]
    assert (select.index("self._check_prefill_budget(")
            < select.index('"prefill_budget_accepted" if can_admit else "prefill_budget_rejected"')
            < select.index("if not can_admit:"))
    raw_prefill = scheduler[scheduler.index("    def _get_new_batch_prefill_raw("):
                            scheduler.index("    def update_running_batch(")]
    assert (raw_prefill.index("admitted_before = len(adder.can_run_list)")
            < raw_prefill.index("res = adder.add_one_req(")
            < raw_prefill.index("record_admission_outcome(req, adder, admitted_before, res)")
            < raw_prefill.index("if res != AddReqResult.CONTINUE:"))
    return {"base_commit": commit, "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
            "original_source_hashes_verified": not already_applied,
            "patch_applies": True, "already_applied": already_applied,
            "syntax_checked_files": MANIFEST["changed_files"],
            "guard_precedes_native_auto_ensure": True,
            "native_completion_timing_seams_verified": True,
            "native_dma_ack_observer_seams_verified": True,
            "complete_result_published_before_http_delivery": True,
            "streaming_terminal_publication_seams_verified": True,
            "native_match_observer_after_original_match": True,
            "native_running_offset_function_reused": True,
            "native_request_identity_copied": True,
            "host_admission_precedes_native_allocation_and_dma": True,
            "host_admitted_budget_follows_native_commit_and_pending": True,
            "host_capacity_eviction_observer_verified": True,
            "admission_diagnostic_observes_actual_gate_and_membership": True,
            "local_model_scope_parser_matches_standalone": True,
            "source_only": source_only,
            "gpu_runtime_tested": False, "full_dependency_import_tested": False,
            "host_quote_supported": True,
            "limitation": "Static compilation and CPU contract tests do not validate CUDA execution, IPC startup, performance or quality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=ROOT / "vendor/sglang")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = prepare(args.checkout.resolve(), args.source_only)
    rendered = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
