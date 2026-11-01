"""Summarize observed response-to-next-request gaps in BurstGPT v2 traces."""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random


QUANTILES = (0, .1, .25, .5, .75, .9, .95, .99, 1)
FIELDS = {'Timestamp', 'Session ID', 'Elapsed time', 'Model', 'Response tokens', 'Log Type'}


def _distribution(values, unit, *, short_gap=None):
    ordered = sorted(values)
    if not ordered:
        return {'count': 0}
    quantiles = {}
    for q in QUANTILES:
        position = (len(ordered) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        quantiles[str(q)] = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    result = {'count': len(ordered), f'mean_{unit}': sum(ordered) / len(ordered),
              f'quantiles_{unit}': quantiles}
    if short_gap is not None:
        result['short_gap_threshold_seconds'] = short_gap
        result['short_gap_fraction'] = sum(value <= short_gap for value in ordered) / len(ordered)
        result['over_one_hour_fraction'] = sum(value > 3600 for value in ordered) / len(ordered)
    return result


def _analyze_trace(path: str | Path, *, short_gap_seconds: float = 12):
    if not math.isfinite(short_gap_seconds) or short_gap_seconds < 0:
        raise ValueError('short-gap threshold must be a finite nonnegative number')
    source = Path(path)
    gaps = array('d')
    model_gaps = {'ChatGPT': array('d'), 'GPT-4': array('d')}
    counts = Counter()
    # Retaining the last event, including failures, prevents invented adjacency.
    last = {}
    last_global = float('-inf')
    with source.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not FIELDS <= set(reader.fieldnames or ()):
            raise ValueError(f'unexpected schema: {reader.fieldnames}')
        for row in reader:
            counts['all_rows'] += 1
            try:
                now = float(row['Timestamp'])
            except ValueError:
                raise ValueError(f'invalid timestamp at row {counts["all_rows"]}') from None
            if not math.isfinite(now) or now < last_global:
                raise ValueError(f'trace is not globally ordered at row {counts["all_rows"]}')
            last_global = now
            sid = row['Session ID'].strip()
            if row['Log Type'] != 'Conversation log' or not sid:
                counts['nonconversation_or_missing_session'] += 1
                continue
            try:
                elapsed = float(row['Elapsed time'])
                output = int(row['Response tokens'])
            except ValueError:
                counts['invalid_elapsed_or_output'] += 1
                last.pop(sid, None)
                continue
            if not math.isfinite(elapsed) or elapsed < 0 or output < 0:
                counts['invalid_elapsed_or_output'] += 1
                last.pop(sid, None)
                continue
            model = row['Model']
            counts['conversation_rows'] += 1
            previous = last.get(sid)
            ambiguous = False
            length = 1 if previous is None else previous[4] + 1
            if previous is not None:
                previous_time, previous_elapsed, previous_output, previous_model, _, tied = previous
                if now == previous_time:
                    counts['tied_timestamp_pairs'] += 1
                    ambiguous = True
                elif tied:
                    counts['pairs_after_tied_timestamp'] += 1
                elif previous_output == 0 or output == 0:
                    counts['pairs_touching_failure'] += 1
                else:
                    gap = now - previous_time - previous_elapsed
                    if gap < 0:
                        counts['overlapping_pairs'] += 1
                    else:
                        gaps.append(gap)
                        counts['nonnegative_pairs'] += 1
                        if model == previous_model and model in model_gaps:
                            model_gaps[model].append(gap)
                        elif model != previous_model:
                            counts['cross_model_pairs'] += 1
            last[sid] = (now, elapsed, output, model, length, ambiguous)
    with source.open('rb') as stream:
        sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    summary = {'source': 'https://github.com/HPMLL/BurstGPT/releases/tag/v2.0',
            'file': source.name, 'sha256': sha256,
            'scope': 'next submission minus previous submission and previous elapsed time; '
                     'adjacent same-session conversation rows with nonzero responses; '
                     'overlapping and ambiguous ties excluded and counted; no text or QA quality',
            'counts': dict(counts), 'sessions_retained': len(last),
            'session_observed_rows': _distribution((item[4] for item in last.values()), 'rows'),
            'gap_seconds': _distribution(gaps, 'seconds', short_gap=short_gap_seconds),
            'same_model_gap_seconds': {model: _distribution(values, 'seconds', short_gap=short_gap_seconds)
                                       for model, values in model_gaps.items()}}
    return summary, gaps


def summarize_trace(path: str | Path, *, short_gap_seconds: float = 12) -> dict:
    """Use adjacent observed events; retain failures and ambiguous ties as exclusions."""
    return _analyze_trace(path, short_gap_seconds=short_gap_seconds)[0]


def sample_plan(rows, gaps, *, trace_sha256: str, seed: int = 42, max_gap_seconds=None):
    """Sample individual observed gaps; this does not preserve trace arrival correlations."""
    if not gaps:
        raise ValueError('trace contains no usable gaps')
    if max_gap_seconds is not None and (not math.isfinite(max_gap_seconds) or max_gap_seconds < 0):
        raise ValueError('maximum gap must be finite and nonnegative')
    sessions = {}
    for row in rows:
        sessions.setdefault(str(row['conversation_id']), []).append(row)
    ordered = [(sid, sorted(turns, key=lambda row: int(row['turn']))) for sid, turns in sorted(sessions.items())]
    tasks = [str(row['task_id']) for _, turns in ordered for row in turns[:-1]]
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError('target cohort needs unique non-final task IDs')
    rng = random.Random(seed)
    selected = [float(gaps[rng.randrange(len(gaps))]) for _ in tasks]
    used = [min(value, max_gap_seconds) if max_gap_seconds is not None else value for value in selected]
    return {'schema_version': 1, 'gaps_after_task': dict(zip(tasks, used, strict=True)),
            'source': {'dataset': 'BurstGPT_3.csv', 'trace_sha256': trace_sha256, 'seed': seed,
                       'method': 'independent observed gaps sampled with replacement',
                       'max_gap_seconds': max_gap_seconds,
                       'capped_gaps': sum(a != b for a, b in zip(selected, used, strict=True)),
                       'source_gaps': len(gaps), 'target_sessions': len(sessions)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--short-gap-seconds', type=float, default=12)
    parser.add_argument('--tasks', type=Path, help='benchmark JSONL; only session, turn and task IDs are used')
    parser.add_argument('--split-manifest', type=Path, help='optional target conversation split')
    parser.add_argument('--split', choices=('dev', 'test'), default='test')
    parser.add_argument('--domain', choices=('all', 'cloud'), default='all')
    parser.add_argument('--plan-output', type=Path, help='write a per-task sampled gap plan')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-gap-seconds', type=float, help='explicitly cap sampled gaps for a bounded run')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.tasks is None) != (args.plan_output is None):
        parser.error('--tasks and --plan-output must be supplied together')
    if args.split_manifest is not None and args.tasks is None:
        parser.error('--split-manifest requires --tasks')
    if args.max_gap_seconds is not None and args.tasks is None:
        parser.error('--max-gap-seconds requires --tasks')
    if args.plan_output is not None and args.plan_output.resolve() == args.output.resolve():
        parser.error('--output and --plan-output must differ')
    if args.plan_output is not None and args.plan_output.exists():
        raise FileExistsError(args.plan_output)
    result, gaps = _analyze_trace(args.input, short_gap_seconds=args.short_gap_seconds)
    plan = None
    if args.tasks is not None:
        from adapters.mtrag import CLOUD_COLLECTION, load_rows
        rows = load_rows(args.tasks)
        if args.domain == 'cloud':
            rows = [row for row in rows if row['Collection'] == CLOUD_COLLECTION]
        if args.split_manifest is not None:
            selected = set(json.loads(args.split_manifest.read_text())[args.split])
            rows = [row for row in rows if row['conversation_id'] in selected]
        plan = sample_plan(rows, gaps, trace_sha256=result['sha256'], seed=args.seed,
                           max_gap_seconds=args.max_gap_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if plan is not None:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(json.dumps(plan, indent=2) + '\n')


if __name__ == '__main__':
    main()
