"""Four source-family-isolated controlled calibration tasks; real Qwen/RB_llm labels.

Uses the existing branch collector and scorer. Warm executions supply the
cache fixture; current claims and source coverage come from this task.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from itertools import combinations
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import urlsplit

from controller.action_proposals import build_observed_action_plans, finite_action_domain
from controller.context_schema import ContextBudget, EvidenceSet, EvidenceUnit, Need, ReadingOutcome, SourceSpan, StepRecord, TaskState
from controller.evidence import make_evidence
from controller.plan_gain_protocol import input_protocol
from controller.plan_policy import CompleteCost
from controller.reading import build_read_plan
from controller.resident_models import evidence_selection_contract, resident_model_digest
from controller.rolling_policy import ZERO_AUX, compile_step, namespace_plan
from controller.schema import OnlineContext, SessionSnapshot
from controller.serialization import to_jsonable
from training.plan_gain_completion import finalize, write_rows
from training.plan_gain_protocol import observed_state_binding
from training.plan_gain_rollout import (GenerationBudget, JudgeJournal, ResourceWindows, bounded_engine,
    digest, execute_branch, radbench_scorer_identity, score_branch, write)


CASES = (('train', 'amberharbor', 'AH-47', 'North Services'),
         ('train', 'cedarridge', 'CR-82', 'West Dispatch'),
         ('dev', 'violetbay', 'VB-29', 'East Operations'),
         ('dev', 'silverplain', 'SP-63', 'South Intake'))


def replace_source_catalogue(path, values):
    """Atomically replace only this invocation's generated cumulative catalogue."""
    with tempfile.TemporaryDirectory(prefix='.source-catalogue-', dir=path.parent) as directory:
        temporary = Path(directory)/'sources.jsonl'
        write_rows(temporary, values)
        os.replace(temporary, path)


def completed_prior_tasks(prior, cohort, domain, asset):
    """Reuse completed, scored tasks from the previous collection."""
    if prior is None:
        return {}, {}, None
    prior = Path(prior).resolve()
    if json.loads((prior/'cohort.json').read_text()) != cohort:
        raise ValueError('resume source belongs to a different declared controlled cohort')
    binding = json.loads((prior/'binding.json').read_text())
    if binding['action_domain'] != domain or binding['reader_profile_id'] != asset['reader_profile_id']:
        raise ValueError('resume source changed its action domain or reader profile')
    latest = {}
    for line in (prior/'all-model-calls.jsonl').read_text().splitlines():
        row = json.loads(line)
        latest[row['index']] = row
    complete = {}
    for number, case in enumerate(CASES):
        directory = prior/'tasks'/f'{number:05d}'
        paths = sorted(directory.glob('branch-*/run.json'))
        if not paths:
            continue
        runs = [json.loads(path.read_text()) for path in paths]
        if (len(runs) != 3 or sorted(run['semantic_kind'] for run in runs) != ['direct_answer', 'read_once', 'read_once']
                or any(run['status'] != 'completed' or run['scoring']['status'] != 'scored'
                       or run['metrics'].get('plan_gain_quality') is None or run['qa_increment'] != 1 for run in runs)):
            raise ValueError('resume-from only accepts wholly completed/scored tasks; preserve partial branches for separate recovery')
        registered = json.loads((directory/'source-registration.json').read_text())
        expected, _ = supplied_case(*case)
        if registered['task'] != expected or any(run['task_id'] != expected['task_id'] for run in runs):
            raise ValueError('resume task/reference/source identity differs from the current controlled case')
        complete[number] = {'directory': directory, 'registration': registered, 'runs': runs, 'paths': paths}
    provenance = {'prior_output': str(prior), 'prior_capture_entry': str(prior/'capture-entry.json'),
        'prior_call_journal': str(prior/'all-model-calls.jsonl'), 'prior_paid_call_count': len(latest),
        'prior_paid_calls': list(latest.values()), 'reused_task_indices': sorted(complete),
        'accounting_scope': 'current capture window excludes prior work; prior payments remain in referenced original artifacts'}
    return complete, provenance, binding


def supplied_case(split, station, code, office):
    family = f'controlled-regroup-20261109-{split}-{station}'
    question = f'What are the entry code and assigned office for {station} station? Give both with source citations.'
    details = (f'The entry code for {station} station is {code}.',
               f'The assigned office for {station} station is {office}.',
               f'{station} station opens its archive reading room on Tuesdays.',
               f'{station} station accepts paper correspondence but does not publish personal telephone numbers.')
    sources = []
    for index, detail in enumerate(details):
        # One canonical window per source. Padding is original controlled source
        # prose, unrelated to the requested fields, and declared before any run.
        body = (detail+'\nAdministrative background: '+
            (f'The {station} archive maintains a dated public catalogue of routine local notices. '*45))
        identity = family+f':document-{index}'
        sources.append(OnlineContext(identity, body, 'controlled://'+identity, f'{station} record {index}'))
    return {'task_id': family+':turn-1', 'conversation_id': family, 'turn': 1, 'split': split,
        'corpus_id': 'controlled-regroup-20261109',
        'snapshot_id': family+':sources:'+digest(sources)[:16],
        'cohort_split': split, 'Collection': 'controlled_regroup_calibration', 'question': question,
        'reference_answer': f'The entry code is {code}; the assigned office is {office}.',
        'actual_history': [], 'rollout_policy': 'controlled_real_qwen_empty_history',
        'source_document_family': family}, tuple(sources)


def make_plan(row, contexts, scores, template, snapshot, domain):
    ranked = sorted(zip(contexts, scores, strict=True), key=lambda item: (-float(item[1]), item[0].document_id))
    blocks, units, catalogue = [], [], []
    for label, (source, score) in zip(('A', 'C', 'B', 'D'), ranked, strict=True):
        block = make_evidence(source.text, source.document_id, corpus_id='controlled-regroup-20261109',
            source_uri=source.source, title=source.title)
        tokens = len(template.tokenizer.encode(source.text, add_special_tokens=False))
        if tokens > 2048:
            raise ValueError('controlled source exceeds the existing canonical-window granularity')
        span = SourceSpan(row['task_id']+':'+label, block.key, 0, len(block.text))
        units.append(EvidenceUnit(label, (span,), block.text, body_tokens=tokens,
            retrieval_score=float(score), need_ids=('lookup',), document_family=row['source_document_family']))
        blocks.append(block)
        catalogue.append({'source_id': block.key.passage_id, 'document_family': row['source_document_family'],
            'version': block.key.version, 'text': block.text, 'source_uri': block.source_uri,
            'split': row['split'], 'controlled_source': True, 'retrieval_score': float(score)})
    state = TaskState(row['task_id'], row['question'], needs=(Need('lookup', operation='lookup'),),
        current_turn_id=1, source_scope=tuple(block.key for block in blocks))
    selected = EvidenceSet(row['task_id']+':ranked', tuple(units), source_blocks=tuple(blocks))
    budget = ContextBudget(window_tokens=32768, body_tokens=8192, output_tokens=1024,
        read_input_tokens=20000, read_output_tokens=max(domain['output_budgets']))
    initial = build_read_plan(state, selected, 'selected_sources', budget, template=template,
        snapshot=snapshot, read_layout='source_first', read_packing='source_windows',
        output_tokens_per_fact=domain['output_tokens_per_fact'])
    from controller.request_compiler import RequestCompiler
    compiler = RequestCompiler(template)
    task = initial.steps[0]
    def size(group):
        current = replace(task, input_unit_ids=tuple(unit.unit_id for unit in group))
        return len(compiler.compile_read(initial, current, group, (), snapshot,
                                        outputs=(task.output_budget,))[0].input_ids)
    pair = max(size(group) for group in combinations(units, 2))
    triple = min(size(group) for group in combinations(units, 3))
    if triple-pair < 64:
        raise ValueError('source geometry leaves no robust two-window packing interval')
    cap = (pair+triple)//2
    plan = build_read_plan(state, selected, 'selected_sources', replace(budget, read_input_tokens=cap),
        template=template, snapshot=snapshot, read_layout='source_first', read_packing='source_windows',
        output_tokens_per_fact=domain['output_tokens_per_fact'])
    if tuple(frozenset(task.input_unit_ids) for task in plan.steps) != (frozenset(('A', 'C')), frozenset(('B', 'D'))):
        raise ValueError('actual tokenizer/packer did not produce the declared AC/BD baseline')
    return namespace_plan(plan), catalogue, {'max_pair_tokens': pair, 'min_triple_tokens': triple, 'read_input_cap': cap}


async def warm_source(service, plan, snapshot, directory):
    from profiling.common import prepare_cache_fixture
    compiler, engine = service.request_compiler, service.engine
    await prepare_cache_fixture(engine, 'cold')
    info = await engine.info()
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    task = replace(plan.steps[0], step_id='warm:'+plan.task_state.task_id, input_unit_ids=('A', 'B'))
    candidate = compiler.compile_read(plan, task, (units['A'], units['B']), (), snapshot,
        outputs=(task.output_budget,), cache_scope=info)[0]
    write(directory/'warm-registration.json', {'task': task, 'compiled_input': candidate,
        'purpose': 'actual cache-path warmup only; no claims or coverage imported into the current task',
        'actual_max_new_tokens': 1})
    identity = 'warm:'+plan.task_state.task_id
    epoch = await engine.prepare(identity, identity+':prepare', info.engine_restart_id)
    try:
        result = await engine.generate(input_ids=candidate.input_ids, session_id=identity,
            expected_generation=epoch.session_generation, expected_restart_id=info.engine_restart_id,
            attempt_id=identity+':generate', fingerprint=compiler.template.fingerprint, max_new_tokens=1)
    finally:
        await engine.retire(identity, epoch.session_generation, identity+':retire', info.engine_restart_id, 'cache witness')
    view = compiler.observe_executed(candidate, result.receipt,
        cache_config_hash=info.cache_config_hash, engine_restart_id=info.engine_restart_id)
    record = StepRecord(identity, task.step_id, task, input_messages=candidate.messages,
        input_ids=candidate.input_ids, output_ids=result.output_ids, receipt=result.receipt,
        metrics={'source_view': view, 'cache_config_hash': info.cache_config_hash, 'cache_warm_only': True})
    write(directory/'warm-execution.json', record)
    return record


async def collect(args):
    import httpx
    from adapters.hybrid_retrieval import LocalBGE
    from bench.official import load_radbench_contract
    from bench.study_resources import StudyResourceSampler, listening_processes
    from controller.config import create_controller, create_model_template, load_config
    from profiling.retention_capture_cost import CaptureRunAccounting

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    asset = json.loads(Path(args.base_asset).read_text())
    prior_domain = asset['provenance']['action_domain']
    domain = finite_action_domain(prior_domain['output_budgets'], prior_domain['output_tokens_per_fact'])
    gaps = asset['provenance']['continuation_gap_limits']
    cohort = {'kind': 'controlled_calibration_only_not_formal_dev350', 'members': [
        {'task_id': supplied_case(*case)[0]['task_id'], 'conversation_id': supplied_case(*case)[0]['conversation_id'],
         'turn': 1, 'split': case[0], 'source_document_family': supplied_case(*case)[0]['source_document_family']}
        for case in CASES]}
    write(output/'cohort.json', cohort)
    cohort_hash = digest(cohort)
    reused, resume, prior_binding = completed_prior_tasks(args.resume_from, cohort, domain, asset)
    write(output/'resume-provenance.json', resume)
    run = load_config(args.run_config)
    run = replace(run, state_dir=str(output/'state'), plan_gain=None, state_observation=None,
        cache_compilation={'mode': 'contextpilot', 'checkout': args.contextpilot_checkout},
        prefix_retention={'mode': 'native'}, engine=replace(run.engine, base_url=args.reader_url),
        planner=replace(run.planner, strategy='always_rebuild'))
    judge_run = load_config(args.judge_config or args.run_config)
    roots = tuple(sorted(set((*listening_processes((urlsplit(args.reader_url).port,)), os.getpid()))))
    judge_roots = tuple(sorted(set((*listening_processes((urlsplit(args.judge_url).port,)), os.getpid()))))
    resources = ResourceWindows(StudyResourceSampler((args.gpu,), roots, interval=.2),
                               StudyResourceSampler((args.judge_gpu,), judge_roots, interval=.2))
    class ContinuationAccounting(CaptureRunAccounting):
        async def __aexit__(self, error_type, error, traceback):
            result = await super().__aexit__(error_type, error, traceback)
            path = self.output/'capture-entry.json'
            entry = json.loads(path.read_text())
            entry['resume'] = resume
            write(path, entry)
            return result

    accounting = ContinuationAccounting(run, output, args.gpu,
        resource_gpu_devices=tuple(sorted({args.gpu, args.judge_gpu})), resource_process_roots=tuple(sorted(set((*roots, *judge_roots)))))
    remaining_calls = args.max_generations-resume.get('prior_paid_call_count', 0)
    if remaining_calls < 1 and len(reused) < len(CASES):
        raise ValueError('prior paid calls exhausted the declared total generation allocation')
    budget = GenerationBudget(max(0, remaining_calls), output/'all-model-calls.jsonl')
    contract = load_radbench_contract(args.official_checkout)
    service = reranker = client = None
    sources, branches = [], []
    async with accounting:
        await resources.start()
        try:
            service = create_controller(run)
            info = await service.engine.info()
            if info.engine_kind != 'sglang' or info.fingerprint.model_revision != asset['reader_checkpoint']:
                raise ValueError('calibration requires the actual fixed Qwen3-32B reader from the base asset')
            actual_protocol = input_protocol(service.template.fingerprint)
            reference_protocol = asset['provenance']['input_protocol']
            if actual_protocol != reference_protocol:
                raise ValueError('base asset and reader input protocols differ')
            service.engine = bounded_engine(service.engine, budget, output/'reader-wire.json')
            accounting.calls = service.engine.calls
            reranker = await asyncio.to_thread(LocalBGE, args.reranker_model, reranker=True,
                device='cpu', dtype='float32', batch_size=4)
            manifest = {'selection_mode': 'ranked', 'evidence_selection_order': 'published_reranker_dependency_closure_v1',
                'retrieval': {'corpus_id': 'controlled-regroup-20261109',
                    'retriever': {'kind': 'controlled_supplied_pool', 'reranker': reranker.manifest}},
                'source_scope': 'four declared originals per family; no gold-dependent selection or support labels'}
            if evidence_selection_contract(manifest) != asset['provenance']['evidence_selection_contract']:
                raise ValueError('controlled source ranking does not match the base selection contract')
            accounting.resident_model_manifest = manifest
            judge_template = create_model_template(judge_run.model)
            client = httpx.AsyncClient(base_url=args.judge_url, timeout=600)
            response = await client.post('/context/engine/info', json={})
            response.raise_for_status()
            judge_info = response.json()
            if judge_info['fingerprint'] != to_jsonable(judge_template.fingerprint):
                raise ValueError('actual RB_llm judge and configured tokenizer differ')
            scorer = radbench_scorer_identity(contract, judge_info['fingerprint'])
            if prior_binding is not None and (prior_binding['scorer_identity'] != scorer
                    or prior_binding['reader_engine_info']['fingerprint'] != to_jsonable(info.fingerprint)
                    or prior_binding['actual_input_protocol'] != actual_protocol):
                raise ValueError('resume changed its actual reader/input/scorer identity')
            judge = JudgeJournal(client, budget, output/'judge-wire.jsonl')
            write(output/'binding.json', {'reader_engine_info': info, 'judge_engine_info': judge_info,
                'scorer_identity': scorer, 'resident_model_manifest': manifest, 'action_domain': domain,
                'reader_profile_id': asset['reader_profile_id'], 'arguments': vars(args), 'cohort': cohort,
                'reference_input_protocol': reference_protocol, 'actual_input_protocol': actual_protocol,
                'base_asset_usage': 'reader and action-domain metadata only; no predictions or labels loaded'})
            for number, prior in reused.items():
                directory = output/'tasks'/f'{number:05d}'
                directory.parent.mkdir(parents=True, exist_ok=True)
                directory.symlink_to(prior['directory'], target_is_directory=True)
                sources.extend(prior['registration']['catalogue'])
                branches.extend({'task_id': saved['task_id'], 'plan_id': saved['plan_id'], 'status': saved['status'],
                    'quality': saved['metrics']['plan_gain_quality'], 'run': str(path.resolve()),
                    'reused_from': str(prior['directory'])} for saved, path in zip(prior['runs'], prior['paths'], strict=True))
            if sources:
                replace_source_catalogue(output/'sources.jsonl', sources)
            write(output/'progress.json', {'branches': branches, 'generation_calls': 0, 'resume': resume})
            for number, case in enumerate(CASES):
                if number in reused:
                    continue
                directory = output/'tasks'/f'{number:05d}'
                directory.mkdir(parents=True)
                row, contexts = supplied_case(*case)
                snapshot = SessionSnapshot('controlled:'+row['task_id'], corpus_snapshot_id=row['snapshot_id'])
                rank_start = time.monotonic()
                usage_before = dict(reranker.usage)
                scores = await asyncio.to_thread(reranker.score, row['question'], contexts)
                plan, catalogue, geometry = make_plan(row, contexts, list(scores), service.template, snapshot, domain)
                rank_end = time.monotonic()
                rank_physical = await resources.finish('online', rank_start, rank_end)
                sources.extend(catalogue)
                replace_source_catalogue(output/'sources.jsonl', sources)
                write(directory/'source-registration.json', {'task': row, 'contexts': contexts,
                    'scores': list(map(float, scores)), 'catalogue': catalogue, 'geometry': geometry,
                    'original_plan': plan, 'registered_at': rank_start})
                warm_start = time.monotonic()
                cache_source = await warm_source(service, plan, snapshot, directory)
                warm_end = time.monotonic()
                write(directory/'warm-cost.json', {'scope': 'separate controlled cache setup; excluded from current task paid cost',
                    'start': warm_start, 'end': warm_end, **await resources.finish('online', warm_start, warm_end)})
                decision_start = time.monotonic()
                frozen = ReadingOutcome(plan, auxiliary_usage=dict(ZERO_AUX))
                scope = await service.engine.info()
                plans, _ = build_observed_action_plans(frozen, snapshot, service.template,
                    reader_checkpoint=asset['reader_checkpoint'], read_choices=2,
                    request_compiler=service.request_compiler, cache_scope=scope,
                    output_tokens_per_fact=domain['output_tokens_per_fact'])
                reads = tuple(branch for branch in plans if branch.semantic_kind == 'read_once')
                if len(reads) != 2 or ':regroup:' not in reads[1].context_plan.plan_id:
                    raise ValueError('indexed AB regroup candidate unavailable')
                known = tuple(candidate for branch in reads for step in branch.steps if step.task.kind == 'READ'
                    for candidate in compile_step(branch.context_plan, step.task, frozen, snapshot, service.template,
                        outputs=(step.task.output_budget,), request_compiler=service.request_compiler, cache_scope=scope))
                native = await service.engine.quote_batch(known, service.template.fingerprint)
                decision_end = time.monotonic()
                decision_physical = await resources.finish('online', decision_start, decision_end)
                paid = CompleteCost(((rank_end-rank_start)+(decision_end-decision_start))*1000,
                    (rank_physical['gpu_gib_seconds']+decision_physical['gpu_gib_seconds'])*1000,
                    (rank_physical['host_gib_seconds']+decision_physical['host_gib_seconds'])*1000,
                    reranker.usage['input_tokens']-usage_before['input_tokens'], 0,
                    max(rank_physical['gpu_peak_gib'], decision_physical['gpu_peak_gib']),
                    max(rank_physical['host_peak_gib'], decision_physical['host_peak_gib']))
                observed = observed_state_binding(frozen, snapshot, paid_cost=paid, native_condition=native,
                    reader_checkpoint=asset['reader_checkpoint'], reader_profile_id=asset['reader_profile_id'],
                    paid_observation={'basis': 'measured ranking and pre-action compile/quote; separate warmup excluded',
                        'rank_window': [rank_start, rank_end], 'decision_window': [decision_start, decision_end]})
                observed.update(action_domain=domain, input_protocol=input_protocol(service.template.fingerprint),
                    evidence_selection_contract=evidence_selection_contract(manifest),
                    resident_model_manifest_digest=resident_model_digest(manifest),
                    cohort_split=row['split'], cohort_manifest_sha256=cohort_hash)
                comparison = {'body_tokens': plan.budget.body_tokens, 'answer_output_tokens': plan.budget.output_tokens,
                    'window_tokens': plan.budget.window_tokens, 'read_output_budget_grid': None,
                    'action_protocol': asset['provenance']['continuation_protocol'],
                    'reference_state': 'same frozen original material/actual history; direct has zero new READ'}
                binding = {'reader_checkpoint': asset['reader_checkpoint'], 'reader_profile_id': asset['reader_profile_id'],
                    'scorer_identity': scorer, 'split': row['split'], 'cohort_split': row['split'],
                    'cohort_manifest_sha256': cohort_hash, 'measurement_kind': 'hardware',
                    'resident_model_manifest': manifest, 'input_protocol': input_protocol(service.template.fingerprint),
                    'continuation_protocol': asset['provenance']['continuation_protocol'],
                    'continuation_gap_limits': gaps, 'collection_mode': 'observed_actions',
                    'comparison_budget': comparison, 'controlled_calibration': True}
                preparation = {'shared_cost_id': row['task_id']+':paid-preparation', 'complete_ms': 0,
                    'gpu_gib_seconds': 0, 'host_gib_seconds': 0, 'input_tokens': 0, 'output_tokens': 0,
                    'scope': 'all current source preparation already included in observed paid_cost'}
                write(directory/'observation.json', {'working_state': frozen, 'snapshot': snapshot,
                    'observed_state_binding': observed, 'read_organizations': [branch.context_plan for branch in reads],
                    'cache_source': cache_source})
                for index, branch in enumerate(plans):
                    branch_dir = directory/f'branch-{index:02d}'
                    result = await execute_branch(service, branch, frozen, snapshot, row,
                        case_dir=branch_dir, binding=binding, preparation=preparation, resources=resources,
                        cache_start='paired', generation_budget=budget, observed=observed, cache_source=cache_source)
                    if result is None:
                        raise RuntimeError('generation allocation exhausted before registered branch')
                    if result['status'] == 'completed':
                        before = json.loads((branch_dir/'before.json').read_text())
                        result = await score_branch(judge, judge_template, row, before, result, contract=contract,
                            path=branch_dir/'run.json', identity=scorer, resources=resources,
                            judge_window=judge_run.context.max_total_model_tokens, expected_fingerprint=judge_info['fingerprint'])
                    branches.append({'task_id': row['task_id'], 'plan_id': branch.plan_id, 'status': result['status'],
                        'quality': result['metrics'].get('plan_gain_quality'), 'run': str(branch_dir/'run.json')})
                    write(output/'progress.json', {'branches': branches, 'generation_calls': len(budget.tickets), 'resume': resume})
        finally:
            if reranker is not None:
                reranker.close()
            if service is not None:
                await service.aclose()
            if client is not None:
                await client.aclose()
            await resources.close()
    manifest = finalize(output, output/'finalized')
    records = [json.loads(line) for line in (output/'finalized/effective-gains.jsonl').read_text().splitlines()]
    for split in ('train', 'dev'):
        write_rows(output/f'{split}.jsonl', [record for record in records if record['split'] == split])
    write(output/'manifest.json', {**manifest, 'cohort': cohort, 'branches': branches, 'resume': resume,
        'status': 'completed', 'scope': 'controlled mechanism calibration; not formal queue quality or latency evidence'})
    print(json.dumps({'output': str(output), 'pairs': len(records), 'branches': len(branches)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-config', default='configs/qwen3_32b.yaml')
    parser.add_argument('--judge-config')
    parser.add_argument('--reader-url', default='http://127.0.0.1:30000')
    parser.add_argument('--judge-url', default='http://127.0.0.1:30000')
    parser.add_argument('--base-asset', required=True)
    parser.add_argument('--reranker-model', default='models/bge-reranker-v2-m3')
    parser.add_argument('--contextpilot-checkout', default='vendor/contextpilot')
    parser.add_argument('--official-checkout', default='data/mtrag-official')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--judge-gpu', type=int, default=0)
    parser.add_argument('--max-generations', type=int, default=64)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume-from', help='Read-only prior collection; reuse only whole three-branch scored tasks')
    asyncio.run(collect(parser.parse_args()))


if __name__ == '__main__':
    main()
