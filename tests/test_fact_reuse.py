"""Atomic source reuse, scope coverage and request-local support work."""
from dataclasses import replace
import json
from types import SimpleNamespace

from adapters.evidence_pool import SourceDirectory
from controller.context_schema import (ContextPlan, EvidenceClaim, EvidenceSet, Need, OpenNeed,
    ReadResult, ReadTask, ReadingOutcome, RetrievalState, StepRecord, TaskState, TextRef, VersionIntent)
from controller.evidence import make_evidence
from controller.fact_reuse import (ReadReuse, apply_reuse_to_plan, build_reuse_seed,
                                  complete_reuse_ranges, read_contracts)
from controller.schema import CompletionReceipt, SessionSnapshot, TurnRecord
from controller.templates import ByteTemplate


def source_read(*, operation="sum", quantity=True, bridge=False):
    text = "A is owned by B." if bridge else "Record A: 3 kg.\nRecord B: 4 kg."
    block = make_evidence(text, "register", version="v1")
    directory = SourceDirectory(SimpleNamespace(blocks=(block,), source_metadata={}), None, ByteTemplate().tokenizer)
    units = directory.full_scope((block.key,)).units
    question = "Find its owner." if bridge else "Sum record weights."
    state = TaskState("old", question, needs=(Need("old:n", (TextRef(1, "user", 0, len(question)),),
                                                   operation=operation),), current_turn_id=1)
    atoms = tuple(EvidenceClaim(f"read:0:claim:{index}", "old:n", "extract", text=text,
        entity=entity, relation="owner" if bridge else "weight", value="B" if bridge else value if quantity else None,
        unit="" if bridge or not quantity else "kg", source_spans=units[0].source_spans, version="v1")
        for index, (entity, value) in enumerate((("A", "3"),) if bridge else (("A", "3"), ("B", "4"))))
    gap = (OpenNeed("old:n", "headquarters", ("B",), "current", (atoms[0].claim_id,)),) if bridge else ()
    task = ReadTask("read:0", "READ", ("old:n",), tuple(unit.unit_id for unit in units))
    result = ReadResult(task.step_id, atoms, task.input_unit_ids, () if bridge else ("old:n",), open_needs=gap)
    receipt = CompletionReceipt("rid", "old-attempt", "engine", 1, "restart", "hash", "stop", True)
    record = StepRecord("old-attempt", task.step_id, task, result=result, receipt=receipt,
                        input_ids=(1, 2), output_ids=(3,))
    old_sum = EvidenceClaim("reduce:old", "old:n", "derived", value="999",
                           parent_claim_ids=tuple(atom.claim_id for atom in atoms), operation="sum")
    plan = ContextPlan("old-plan", state, EvidenceSet("old-set", units, source_blocks=(block,)), steps=(task,))
    outcome = ReadingOutcome(plan, (result,), (record,), (*atoms, old_sum))
    memory = RetrievalState(1, (TurnRecord(1, question, "An actual answer."),), (state,),
                            source_index=directory, committed_readings=(("old-attempt", outcome),))
    return memory, read_contracts(memory), block


def test_atomic_reuse_shares_facts_recomputes_views_and_never_copies_old_costs():
    from controller.reading import advance_read_plan, build_read_plan, numeric_selection_task
    memory, contracts, _ = source_read()
    needs = (Need("count", operation="count"), Need("sum", operation="sum"))
    plan = ContextPlan("now", TaskState("now", "Count and sum records.", needs=needs), EvidenceSet("empty", ()))
    seed = build_reuse_seed(plan, tuple(ReadReuse(need.need_id, contracts[0].read_id, "complete") for need in needs),
                           contracts, memory)
    assert len(seed.claims) == 2 and all(claim.kind == "extract" for claim in seed.claims)
    assert all(claim.serves("count") and claim.serves("sum") for claim in seed.claims)
    assert all(claim.claim_id.startswith("history:old-attempt:") for claim in seed.claims)
    assert not seed.steps and all(not result.input_tokens and not result.output_tokens for result in seed.results)
    assert set(complete_reuse_ranges(seed)) == {"count", "sum"}
    assert all(step.step_id.startswith("policy:reuse:") for step in seed.plan.steps)
    assert numeric_selection_task(seed.plan, seed.results, need_id="sum") is not None
    assert all(step.operation == "numeric_select" for step in advance_read_plan(seed.plan, seed.results))
    partial = build_reuse_seed(plan, (ReadReuse("sum", contracts[0].read_id, "facts"),), contracts, memory)
    assert partial.claims and not partial.coverage and complete_reuse_ranges(partial) == {"sum": ()}
    current = tuple(replace(unit, unit_id="fresh:"+unit.unit_id,
        source_spans=tuple(replace(span, span_id="fresh:"+span.span_id) for span in unit.source_spans))
        for unit in contracts[0].units)
    physical = build_read_plan(plan.task_state, EvidenceSet('fresh', current,
        source_blocks=tuple(memory.source_index._blocks.values())), 'all_supplied_documents', plan.budget,
        template=ByteTemplate(), read_packing='source_windows')
    applied = apply_reuse_to_plan(physical, seed, ByteTemplate(), SessionSnapshot('s'))
    assert applied.plan.coverage_units == physical.coverage_units
    assert {task.step_id for task in applied.plan.steps} == {result.step_id for result in seed.results}
    assert all(task.operation == 'numeric_select' for task in advance_read_plan(applied.plan, applied.results))
    sum_seed = build_reuse_seed(physical, (ReadReuse('sum', contracts[0].read_id, 'complete'),), contracts, memory)
    count_only = apply_reuse_to_plan(physical, sum_seed, ByteTemplate(), SessionSnapshot('s'))
    pending = tuple(task for task in count_only.plan.steps if task.step_id not in {r.step_id for r in count_only.results})
    assert pending and all(task.need_ids == ('count',) for task in pending)
    facts_only = apply_reuse_to_plan(physical, partial, ByteTemplate(), SessionSnapshot('s'))
    pending = tuple(task for task in facts_only.plan.steps if task.step_id not in {r.step_id for r in facts_only.results})
    assert pending and all(task.need_ids == ('count', 'sum') for task in pending)
    switched = replace(plan, task_state=replace(plan.task_state, version_intent=VersionIntent("explicit", version_ids=("v2",))))
    assert not build_reuse_seed(switched, (ReadReuse("sum", contracts[0].read_id, "complete"),), contracts, memory).claims


def test_reused_bridge_preserves_gap_and_teacher_binds_only_visible_contracts():
    import httpx
    from jsonschema import Draft202012Validator
    from adapters.history_state import TaskInterpreter, TeacherTaskModel
    from adapters.query_selector import OpenAICompletionClient, SelectorConfig
    from controller.schema import OnlineTurn

    memory, contracts, _ = source_read(operation="infer", bridge=True)
    class Tokenizer:
        chat_template = "test"
        def __call__(self, text, **kwargs):
            return {"input_ids": list(text.encode())}
        def apply_chat_template(self, messages, **kwargs):
            return list("".join(message['content'] for message in messages).encode())
    observed = {}
    def respond(request):
        body = json.loads(request.content)
        visible = json.loads(body['messages'][-1]['content'])
        observed.update(payload=visible, schema=body['response_format']['json_schema']['schema'])
        assert [item['read_id'] for item in visible['read_contracts']] == ['r0']
        value = {'relation': 'standalone', 'resume_task_id': '', 'history_spans': [], 'conditions': {},
            'condition_removals': [], 'version_intent': {'mode': 'current', 'temporal_relation': 'at',
                'document_families': [], 'version_ids': [], 'resolved_version_ids': [], 'time_refs': [], 'source_turns': []},
            'needs': {'n0': {'anchor_spans': [{'window_id': 'w0', 'boundary_tokens': [0, 0]}],
                'subject_refs': [], 'operation': 'infer', 'condition_ids': [], 'predecessor_need_ids': [],
                'scope': 'selected_sources', 'read_reuse': {'r0': 'complete'}}}}
        Draft202012Validator(observed['schema']).validate(value)
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(value)}, 'finish_reason': 'stop'}],
                                        'usage': {'completion_tokens': 100}})
    client = OpenAICompletionClient(SelectorConfig('http://task-fixture', 'fixture', 'fixture',
        max_input_tokens=16000, max_new_tokens=2048), tokenizer=Tokenizer(), transport=httpx.MockTransport(respond))
    teacher = TeacherTaskModel(client)
    index = SimpleNamespace(tokenizer=client.tokenizer, select=lambda *args, **kwargs: ())
    interpreter = TaskInterpreter(teacher, index)
    try:
        state = interpreter.interpret(OnlineTurn('s', 'now', 2, 'corpus', "Where is its owner's headquarters?", (), (), 'P3'), memory)
        assert client.calls == 1
    finally:
        client.close()
    seed = build_reuse_seed(ContextPlan('now', state, EvidenceSet('empty', ())),
        interpreter.last_prediction.read_reuse, interpreter.last_read_contracts, memory)
    assert seed.open_needs[0].need_id == 'now:n0'
    assert seed.open_needs[0].bound_entities == ('B',)
    assert seed.open_needs[0].parent_claim_ids == (seed.claims[0].claim_id,)
    assert contracts[0].read_id == interpreter.last_prediction.read_reuse[0].read_id
    contract_ids = observed['schema']['$defs']['Need_0_0_n0']['properties']['read_reuse']
    assert set(contract_ids['properties']) == {'r0'} and contract_ids['additionalProperties'] is False


def test_trained_support_memo_restores_the_matching_query_embedding():
    from adapters.support_model import SupportInput, SupportPrediction
    from training.support_model import EmbeddingCache, TrainedSupportBackend
    memory, contracts, _ = source_read()
    unit = contracts[0].units[0]
    backend = object.__new__(TrainedSupportBackend)
    backend.embeddings, backend._predictions, backend.usage = EmbeddingCache(1), {}, {'cache_hits': 0}
    calls = []
    def compute(examples):
        calls.extend(examples)
        for example in examples:
            backend.embeddings[example.task_id, example.need_id, tuple(u.unit_id for u in example.units)] = example.query
        return tuple(SupportPrediction(.8, .8, .8, .4, 'complete') for example in examples)
    backend._predict_uncached = compute
    first = SupportInput('task', 'need', 'Original query', (unit,))
    changed = replace(first, query='New OpenNeed context')
    backend.predict((first,))
    backend.predict((changed,))
    backend.predict((replace(first, units=(replace(unit, retrieval_score=9),)),))
    assert len(calls) == 2 and backend.embeddings['task', 'need', (unit.unit_id,)] == first.query
    backend.release_task('task')
    backend.predict((first,))
    assert len(calls) == 3


def test_teacher_support_memo_only_computes_changed_input_and_is_released_per_qa():
    from adapters.support_model import SupportInput
    from adapters.teacher_support import TeacherSupportBackend
    memory, contracts, _ = source_read()
    unit = contracts[0].units[0]
    backend = TeacherSupportBackend('http://unused-fixture', ByteTemplate())
    calls = []
    def generate(system, payload, schema, kind, output_tokens):
        calls.append(payload)
        return {'predictions': {index: {'applicability': .8, 'direct_support': .8,
            'joint_support': .8, 'bridge_value': .4, 'disposition': 'complete', 'missing_relation': '',
            'span_selection': {identity: True for identity in row['properties']['span_selection']['properties']}}
            for index, row in schema['properties']['predictions']['properties'].items()}}
    backend._generate = generate
    first = SupportInput('task', 'need', 'Question', (unit,))
    try:
        backend.predict((first, first))
        backend.predict((replace(first, units=(replace(unit, retrieval_score=9),)),))
        assert len(calls) == 1 and backend.usage['examples'] == 1
        backend.predict((replace(first, query='Changed OpenNeed'),))
        assert len(calls) == 2
        backend.release_task('task')
        backend.predict((first,))
        assert len(calls) == 3
    finally:
        backend.close()
