"""Export independently observed development READ/REDUCE timing profiles."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from profiling.common import Measurement
from controller.reading import step_generation_kind


def reading_measurements(records, *, source, run_id):
    groups, ledger = {}, []
    for row in records:
        if row["split"] != "dev":
            raise ValueError("reading calibration consumes the frozen development split")
        outcome = row.get("reading_outcome")
        steps = outcome["steps"] if outcome is not None else row.get("steps", ())
        for step in steps:
            task, metrics = step["task"], step["metrics"]
            mode = outcome["plan"]["reading_mode"] if outcome is not None and "plan" in outcome else "typed_reduce"
            kind = metrics.get("generation_kind", step_generation_kind(task["kind"], task["operation"], mode))
            identity = {"task_id": row["task_id"], "step_id": step["step_id"], "kind": kind,
                        "run_id": run_id, "split": "dev"}
            if not step["input_ids"]:
                ledger.append({**identity, "status": "cpu_operation", "metrics": metrics})
                continue
            receipt, quote = step["receipt"], metrics.get("cache_quote")
            if receipt is None or quote is None or quote["active_requests"] is None:
                ledger.append({**identity, "status": "unmeasured_native_features", "metrics": metrics})
                continue
            input_tokens, output_tokens = len(step["input_ids"]), len(step["output_ids"])
            prefix = receipt["actual_device_hit_tokens"]+receipt["actual_host_loaded_tokens"]
            concurrency = quote["active_requests"]+1
            phases = []
            base = {"prompt_tokens": input_tokens, "prefix_tokens": prefix,
                "output_tokens": output_tokens, "concurrency": concurrency, "transfer_bytes": 0,
                "host_load_tokens": receipt["actual_host_loaded_tokens"],
                "source": source, "run_id": run_id,
                "cache_state": "native_observed_"+quote["cache_config_hash"]}
            if (metrics.get("ttft_ms") is None or metrics.get("last_token_ms") is None
                    or metrics.get("step_wall_ms") is None):
                ledger.append({**identity, "status": "unmeasured_client_timing",
                    "cache_quote": quote, "receipt": receipt, "metrics": metrics})
                continue
            phases.append(Measurement(**base, phase="first_token", elapsed_ms=metrics["ttft_ms"],
                timing_scope="client_first_new_token_including_queue_and_transport"))
            if output_tokens > 1:
                phases.append(Measurement(**base, phase="decode",
                    elapsed_ms=metrics["last_token_ms"]-metrics["ttft_ms"],
                    timing_scope="client_first_to_last_observed_token_chunk"))
            phases.append(Measurement(**base, phase="control",
                elapsed_ms=metrics["step_wall_ms"]-metrics["last_token_ms"],
                timing_scope="step_wall_minus_last_observed_token_including_receipt_and_retire"))
            groups.setdefault(kind, []).extend(phases)
            ledger.append({**identity, "status": "observed", "input_tokens": input_tokens,
                "output_tokens": output_tokens, "output_budget": task["output_budget"],
                "active_requests": quote["active_requests"], "queued_requests": quote["queued_requests"],
                "actual_prefix_tokens": prefix, "cache_quote": quote,
                "receipt": receipt, "metrics": metrics,
                "exported_phases": [phase.phase for phase in phases]})
    return groups, ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="Study development JSONL, including failed step records")
    parser.add_argument("--source", choices=("hardware", "mock"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    records = [json.loads(line) for line in Path(args.records).read_text().splitlines() if line.strip()]
    groups, ledger = reading_measurements(records, source=args.source, run_id=args.run_id)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for kind, rows in groups.items():
        (output/f"{kind}.jsonl").write_text("".join(json.dumps(asdict(row))+"\n" for row in rows))
    (output/"step-ledger.jsonl").write_text("".join(json.dumps(row)+"\n" for row in ledger))
    (output/"manifest.json").write_text(json.dumps({"source": args.source, "run_id": args.run_id,
        "split": "dev", "records": str(Path(args.records).resolve()),
        "phase_files": {kind: str((output/f"{kind}.jsonl").resolve()) for kind in groups},
        "steps": len(ledger), "status_counts": {status: sum(row["status"] == status for row in ledger)
            for status in {row["status"] for row in ledger}},
        "concurrency_basis": "observed active requests plus this request; queued requests retained separately",
        "prefix_basis": "actual receipt device hit plus Host load",
        "generation_kind_binding": "separate JSON constrained READ/REDUCE and ordinary summary profiles"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
