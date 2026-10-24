"""Small real-server protocol acceptance run; no cache flush and no performance claim."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import uuid

from controller.config import create_model_template, load_config
from controller.engine_client import EngineError, SGLangEngineClient
from controller.schema import PromptCandidate
from controller.templates import token_hash


async def verify(engine, input_ids, *, max_new_tokens=16):
    info = await engine.info()
    if info.engine_kind != "sglang":
        raise ValueError("this acceptance run requires a real patched SGLang")
    sid, op, attempt = (uuid.uuid4().hex for _ in range(3))
    ack = await engine.prepare(sid, op, info.engine_restart_id)
    assert await engine.prepare(sid, op, info.engine_restart_id) == ack
    candidate = PromptCandidate("smoke", "APPEND", 0, (), tuple(input_ids), token_hash(input_ids),
                                frozenset(), frozenset())
    before = await engine.metrics()
    result = await engine.generate(input_ids=input_ids, session_id=sid,
        expected_generation=ack.session_generation, expected_restart_id=info.engine_restart_id,
        attempt_id=attempt, fingerprint=engine.fingerprint, max_new_tokens=max_new_tokens)
    assert result.generation_complete and result.receipt.session_ref_registered
    assert result.receipt.actual_prefill_tokens > 0
    assert result.server_ttft_ms is not None and result.server_elapsed_ms >= result.server_ttft_ms
    status = await engine.status(attempt_id=attempt, expected_restart_id=info.engine_restart_id)
    assert status.terminal and status.result is not None
    assert (status.result.text, status.result.output_ids, status.result.receipt) == (
        result.text, result.output_ids, result.receipt)
    # Quotes reuse the native match limit; any partial final page stays excluded.
    first = await engine.quote_batch([candidate], engine.fingerprint)
    second = await engine.quote_batch([candidate], engine.fingerprint)
    assert first.quotes == second.quotes
    assert first.quotes[0].ready_gpu_prefix_tokens == first.quotes[0].match_limit
    retired = await engine.retire(sid, ack.session_generation, uuid.uuid4().hex,
                                  info.engine_restart_id, "smoke_complete")
    assert retired.acknowledged
    assert retired.freed_gpu_tokens == retired.freed_host_tokens == 0
    rejected_attempt = uuid.uuid4().hex
    try:
        await engine.generate(input_ids=input_ids, session_id=sid,
            expected_generation=ack.session_generation, expected_restart_id=info.engine_restart_id,
            attempt_id=rejected_attempt, fingerprint=engine.fingerprint, max_new_tokens=1)
    except EngineError:
        pass
    else:
        raise AssertionError("retired epoch admitted another request")
    rejected = await engine.status(attempt_id=rejected_attempt, expected_restart_id=info.engine_restart_id)
    assert rejected.state == "REJECTED" and rejected.terminal
    after = await engine.metrics(before["next_seq"])
    return dict(status="passed", source="hardware", engine=asdict(info), receipt=asdict(result.receipt),
                server_ttft_ms=result.server_ttft_ms, server_elapsed_ms=result.server_elapsed_ms,
                replay_exact=True, stale_epoch_rejected=True, quote_repeat_equal=True,
                retirement_frees_zero_slots=True, metrics_before=before, metrics_after=after,
                limitation="Smoke validates this server/configuration; it is not a throughput or quality result.")


async def _run(args):
    config = load_config(args.config)
    if config.engine.kind != "sglang":
        raise ValueError("engine.kind must be sglang")
    template = create_model_template(config.model, model_path=args.model_path, local_files_only=True)
    nonce = uuid.uuid4().hex
    ids = template.tokenizer.encode(f"ContextKV protocol test {nonce}. "+"Check cache behavior. "*64,
                                    add_special_tokens=False)
    engine = SGLangEngineClient(config.engine.base_url, template.fingerprint,
                                timeout=config.engine.timeout_seconds,
                                **({"model_scope":config.model.scope.manifest()} if config.model.scope else {}))
    try:
        report = await verify(engine, ids, max_new_tokens=args.max_new_tokens)
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({"status":"passed", "report":str(target)}))
    finally:
        await engine.aclose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sglang.yaml")
    parser.add_argument("--model-path", default=os.environ.get("CONTEXTKV_MODEL_PATH"))
    parser.add_argument("--report", default="runs/sglang-smoke.json")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args(argv)
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
