"""Build reproducible calibration grids or summarize actual measured JSONL.

GPU collection is driven by collect(probe, prepare). prepare must explicitly reset
or warm the requested prefix; probe must return real measured phase times, not
prompt-minus-hit estimates. This avoids pretending client wall time is CUDA time.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from profiling.common import Measurement, load_measurements, summarize_measurements, write_measurements


@dataclass(frozen=True)
class PrefillCase:
    prompt_tokens: int
    prefix_tokens: int
    output_tokens: int
    concurrency: int
    cache_state: str
    device_prefix_tokens: int | None = None

    def record(self) -> dict:
        """Keep the original five-field wire shape for legacy fixtures."""
        value = asdict(self)
        if self.device_prefix_tokens is None:
            value.pop("device_prefix_tokens")
        return value

    def prefix_split(self) -> tuple[int, int]:
        """Prepared device/host-only tokens, never inferred from execution."""
        if self.cache_state != "mixed" and self.device_prefix_tokens is not None:
            raise ValueError("device_prefix_tokens is only used by mixed cases")
        if self.cache_state == "cold":
            if self.prefix_tokens:
                raise ValueError("cold cases must request zero reusable prefix")
            return 0, 0
        if self.cache_state not in {"gpu_ready", "host_only", "mixed"}:
            raise ValueError("unknown cache state")
        if self.prefix_tokens <= 0:
            raise ValueError("warm cases need a nonempty reusable prefix")
        if self.cache_state == "mixed":
            device = self.device_prefix_tokens
            if type(device) is not int or not 0 < device < self.prefix_tokens:
                raise ValueError("mixed cases need 0 < device_prefix_tokens < prefix_tokens")
            return device, self.prefix_tokens-device
        return (self.prefix_tokens, 0) if self.cache_state == "gpu_ready" else (0, self.prefix_tokens)


def calibration_grid(lengths: Sequence[int] = (1024, 2048, 4096, 8192, 16384),
                     prefix_fractions: Sequence[float] = (0, .25, .5, .75, .99),
                     outputs: Sequence[int] = (128, 512, 1024), concurrency: Sequence[int] = (1,),
                     page_size: int = 64, context_limit: int = 32768) -> tuple[PrefillCase, ...]:
    if page_size < 1 or context_limit < 2:
        raise ValueError("invalid page size/context limit")
    cases = []
    for n in lengths:
        for fraction in prefix_fractions:
            if not 0 <= fraction <= 1:
                raise ValueError("prefix fraction must be in [0,1]")
            prefix = min(n - 1, int(n * fraction)) // page_size * page_size
            for output in outputs:
                if n < 1 or output < 1 or n + output > context_limit:
                    continue
                for count in concurrency:
                    if count < 1:
                        raise ValueError("concurrency must be positive")
                    cases.append(PrefillCase(n, prefix, output, count, "cold" if prefix == 0 else "gpu_ready"))
    return tuple(cases)


def load_cases(path: str | Path, *, page_size: int, context_limit: int) -> tuple[PrefillCase, ...]:
    """Load selected shapes without generating a Cartesian experiment matrix."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("case file must contain a nonempty JSON list")
    cases = tuple(PrefillCase(**record) for record in records)
    for case in cases:
        if (any(type(value) is not int for value in (case.prompt_tokens, case.prefix_tokens,
                case.output_tokens, case.concurrency)) or min(case.prompt_tokens, case.output_tokens,
                case.concurrency) < 1 or not 0 <= case.prefix_tokens < case.prompt_tokens):
            raise ValueError("case sizes must be integers with positive prompt/output/concurrency and 0 <= prefix < prompt")
        if case.prompt_tokens + case.output_tokens > context_limit:
            raise ValueError("case exceeds configured context limit")
        device, _ = case.prefix_split()
        if case.prefix_tokens % page_size or device % page_size:
            raise ValueError("case prefixes must align to the actual native page size")
    return cases


async def collect(cases: Sequence[PrefillCase], *,
                  prepare: Callable[[PrefillCase], Awaitable[None]],
                  probe: Callable[[PrefillCase], Awaitable[Sequence[Measurement]]],
                  repetitions: int = 3, output_path: str | Path | None = None) -> tuple[Measurement, ...]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rows = []
    for case in cases:
        for _ in range(repetitions):
            await prepare(case)
            measured = tuple(await probe(case))
            if not measured:
                raise ValueError("probe returned no measurements")
            for row in measured:
                if row.source != "hardware":
                    raise ValueError("hardware collection refuses mock measurements")
                if (row.prompt_tokens, row.concurrency) != (case.prompt_tokens, case.concurrency):
                    raise ValueError("probe sizes disagree with requested calibration case")
            if output_path:
                write_measurements(output_path, measured)
            rows.extend(measured)
    return tuple(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", help="Actual Measurement JSONL from an instrumented engine")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-mock", action="store_true")
    args = parser.parse_args(argv)
    if args.measurements:
        payload = {"status": "summarized_observations", "buckets": summarize_measurements(load_measurements(args.measurements), allow_mock=args.allow_mock)}
    else:
        payload = {"status": "not_run", "hardware_measurements": None, "cases": [c.record() for c in calibration_grid()],
                   "instructions": "Use collect(cases, prepare=cache_reset_or_warm, probe=instrumented_engine). No GPU measurements exist in this file."}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
