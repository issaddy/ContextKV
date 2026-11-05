"""No encoder, learning, or model execution: supplied graph logits only."""
import numpy as np
from adapters.history_state import TaskPrediction
from controller.context_schema import Condition, Need, TaskState, TextRef
from training.task_relations import candidates, targets, apply_edges, graph_items


def test_owned_edges_history_scope_and_acyclic_decoding():
    a, b, x, y = [TextRef(1, 'user', i, i+1) for i in range(4)]
    old = Condition('old', value_spans=(x,))
    new = Condition('new', value_spans=(y,))
    previous = Need('previous', (x,))
    state = TaskState('history', '', conditions=(old,), needs=(previous,))
    needs = (Need('a', (a,), subject_refs=(x,), condition_ids=('old',),
                  predecessor_need_ids=('previous',), scope='all_supplied_documents'),
             Need('b', (b,), subject_refs=(y,), condition_ids=('new', 'old')))
    encoded = {'prior_tasks': (state,), 'prior_conditions': (old,)}
    items = candidates(encoded, needs, (new,), (x, y), relation="followup", resume_task_id="history")
    gold = targets(items)
    scores = {name: np.array(values)*4-2 for name, values in gold.items() if name != 'scope'}
    scores['scope'] = np.array([[-2, 2], [2, -2]])
    scores['need_predecessor'][0, 1] = 3
    scores['need_predecessor'][1, 0] = 1
    decoded = apply_edges(items, scores)
    assert decoded[0].subject_refs == (x,)
    assert decoded[1].subject_refs == (y,)
    assert decoded[0].condition_ids == ('old',)
    assert decoded[1].condition_ids == ('new', 'old')
    assert decoded[0].scope == 'all_supplied_documents'
    assert decoded[0].predecessor_need_ids == ('b', 'previous')
    assert decoded[1].predecessor_need_ids == ()
    empty = candidates(encoded, (), (), (), relation="standalone", resume_task_id="")
    assert apply_edges(empty, {'need_predecessor': np.empty((0, 1))}) == ()
    prediction = TaskPrediction('followup', conditions=(new,), needs=needs)
    renamed = TaskPrediction('followup', conditions=(Condition('renamed', value_spans=(y,)),),
        needs=(needs[0], Need('other', (b,), subject_refs=(y,), condition_ids=('renamed', 'old'))))
    assert graph_items(prediction, (old,), (state,)) == graph_items(renamed, (old,), (state,))


def test_full_decode_uses_group_edges_and_multilabel_catalogue_updates():
    from types import SimpleNamespace
    from training.task_model import encode_task, task_targets, decode_prediction, SPAN_KINDS, UPDATES, OPERATIONS
    from controller.context_schema import RetrievalState, VersionIntent
    from controller.schema import TurnRecord
    class Characters:
        cls_token_id, sep_token_id = 1000, 1001
        def __call__(self, text, **kwargs):
            result = {'input_ids': list(range(len(text)))}
            if kwargs.get('return_offsets_mapping'):
                result['offset_mapping'] = [(i, i+1) for i in range(len(text))]
            return result
    prior_ref = TextRef(1, 'user', 0, 1)
    old = (Condition('old1', value_spans=(prior_ref,)), Condition('old2', value_spans=(prior_ref,)))
    state = TaskState('prior', 'z', conditions=old, needs=(Need('priorneed', (prior_ref,)),), current_turn_id=1)
    memory = RetrievalState(1, (TurnRecord(1, 'z', 'actual'),), (state,))
    turn = SimpleNamespace(turn=2, query='abcdef')
    catalogue = {'manual': [{'version': 'v1', 'title': 'old'}, {'version': 'v2', 'title': 'new'}]}
    encoded = encode_task(Characters(), turn, (), memory, catalogue=catalogue, max_input_tokens=3000)
    a, b, key, value, predicate = [TextRef(2, 'user', i, i+1) for i in range(5)]
    condition = Condition('new', key, (value,), predicate_span=predicate,
                          target_refs=('old1', 'old2'), supersedes=('old2',), update='REPLACE')
    expected = TaskPrediction('followup', resume_task_id='prior', conditions=(condition,),
        needs=(Need('newneed', (a, b), condition_ids=('new',), scope='all_supplied_documents'),),
        version_intent=VersionIntent('compare', document_families=('manual',),
            version_ids=('v1', 'v2'), resolved_version_ids=('v2',), source_turns=(1,)))
    gold = task_targets(encoded, expected)
    length = len(encoded['input_ids'])
    output = {}
    for name, size in [('span', len(SPAN_KINDS)*3), ('history', 2), ('operation', len(OPERATIONS)),
                       ('update', len(UPDATES)), ('condition_relation', 6)]:
        output[name] = np.full((1, length, size), -4.)
    for i, labels in enumerate(gold['span']):
        for channel, label in enumerate(labels):
            if label >= 0:
                output['span'][0, i, channel*3+label] = 4
    for name in ('history', 'operation', 'update', 'condition_relation'):
        for i, label in enumerate(gold[name]):
            output[name][0, i, max(0, label)] = 4
    for name in ('relation', 'version', 'temporal_relation', 'resume'):
        values = np.full((1, 8), -4.)
        values[0, gold[name]] = 4
        output[name] = values
    for name in ('condition_target', 'condition_supersedes'):
        output[name] = np.array(gold[name], dtype=float)[None] * 8 - 4
    for name in ('document_families', 'version_ids', 'resolved_version_ids', 'source_turns'):
        output[name] = np.array(gold[name])*8-4
    output['group_scores'] = lambda name, refs: np.ones((len(refs), len(refs)))
    def graph_scores(items):
        values = {name: np.full((len(items['needs']), len(items[kind])), -4.)
                  for name, kind in [('need_subject', 'subjects'), ('need_condition', 'conditions'),
                                     ('need_predecessor', 'predecessors')]}
        values['need_condition'][:, 0] = 4
        values['scope'] = np.array([[-4, 4]])
        return values
    output['graph_scores'] = graph_scores
    decoded = decode_prediction(encoded, output, turn)
    assert decoded.needs[0].anchor_spans == (a, b)
    assert len(decoded.needs) == 1
    assert decoded.conditions[0].key_span == key
    assert decoded.conditions[0].predicate_span == predicate
    assert decoded.conditions[0].value_spans == (value,)
    assert decoded.conditions[0].target_refs == ('old1', 'old2')
    assert decoded.conditions[0].supersedes == ('old2',)
    assert decoded.version_intent == expected.version_intent
    assert decoded.needs[0].scope == 'all_supplied_documents'
    assert graph_items(decoded, old, (state,)) == graph_items(expected, old, (state,))
    # A smaller budget may remove directory/history candidates; the question stays visible.
    from adapters.history_state import HistoryWindow
    required = len(encoded['input_ids'])
    with_optional = encode_task(Characters(), turn,
        (HistoryWindow(TextRef(1, 'assistant', 0, 6), 'actual', 6),), memory,
        catalogue=catalogue, max_input_tokens=required)
    assert len(with_optional['input_ids']) <= required
    assert any(ref is not None and ref.turn_id == turn.turn for ref in with_optional["refs"])


def test_budgeted_history_catalogue_and_invisible_supervision():
    from types import SimpleNamespace
    import pytest
    from training.task_model import encode_task, task_targets, TaskInputNotVisible
    from adapters.history_state import budget_task_catalogue, observed_task_sources, task_source_catalogue
    from controller.context_schema import RetrievalState
    from controller.schema import TurnRecord
    class Characters:
        cls_token_id, sep_token_id = 1000, 1001
        def __call__(self, text, **kwargs):
            return {"input_ids": [1]*len(text), "offset_mapping": [(i, i+1) for i in range(len(text))]}
    tokenizer = Characters()
    history = tuple(TurnRecord(i, "x"*2000, "a") for i in range(1, 5))
    states = tuple(TaskState(str(i), "old", needs=(Need(str(i), (TextRef(i, "user", 0, 2000),)),),
                             current_turn_id=i) for i in range(1, 5))
    turn = SimpleNamespace(turn=5, query="new independent question")
    encoded = encode_task(tokenizer, turn, (), RetrievalState(4, history, states), catalogue={})
    assert len(encoded["input_ids"]) <= 8192
    assert any(ref is not None and ref.turn_id == 5 for ref in encoded["refs"])
    for state in encoded["prior_tasks"]:
        for need in state.needs:
            assert any(owner.start <= need.anchor_spans[0].start and need.anchor_spans[0].end <= owner.end
                       and owner.turn_id == need.anchor_spans[0].turn_id for owner, _ in encoded["text_groups"])
    with pytest.raises(TaskInputNotVisible):
        task_targets(encoded, TaskPrediction("standalone", needs=(Need("old", (TextRef(1, "user", 0, 2000),)),)))
    catalogue = {f"family{i}": [{"version": f"v{i}", "title": "manual"}] for i in range(100)}
    selected = budget_task_catalogue(tokenizer, "family99", catalogue, (), 1024, current_turn_id=5)
    assert "family99" in selected
    assert budget_task_catalogue(tokenizer, "family99", selected, (), 1024, current_turn_id=5) == selected
    small = encode_task(tokenizer, SimpleNamespace(turn=5, query="family99"), (), RetrievalState(0),
                        catalogue=catalogue, max_input_tokens=1024)
    assert len(small["input_ids"]) <= 1024
    assert "v99" in small["catalogue_versions"]
    source = observed_task_sources({"task_catalogue": catalogue, "task_reference_date": "2026-11-04", "source_scope": []})
    assert task_source_catalogue(RetrievalState(0, source_index=source)) == catalogue
    source.observed_source_scope = ("large-scope"*1000,)
    hidden_metadata = encode_task(tokenizer, turn, (), RetrievalState(0, source_index=source), max_input_tokens=256)
    assert hidden_metadata["input_omissions"]
    with pytest.raises(TaskInputNotVisible, match="date/scope"):
        task_targets(hidden_metadata, TaskPrediction("standalone"))


def test_teacher_budgets_complete_input_with_condition_and_predecessor_closure():
    import json
    from types import SimpleNamespace
    import httpx
    from adapters.history_state import HistoryWindow, TeacherTaskModel
    from adapters.query_selector import OpenAICompletionClient, SelectorConfig
    from controller.context_schema import RetrievalState
    from controller.schema import TurnRecord

    class Characters:
        chat_template = 'literal test chat'

        def __call__(self, text, **kwargs):
            return {'input_ids': list(text.encode())}

        def apply_chat_template(self, messages, **kwargs):
            return list(''.join(f'<{item["role"]}>{item["content"]}' for item in messages).encode())

    history, states = [], []
    for index in range(1, 61):
        question = f'Task {index}: count records for ' + f'group-{index} ' * 30
        anchor = question.index('count records')
        value = question.index(f'group-{index}')
        condition = Condition(f'c{index}', value_spans=(TextRef(index, 'user', value, value+len(f'group-{index}')),))
        need = Need(f'n{index}', (TextRef(index, 'user', anchor, anchor+len('count records')),),
            condition_ids=(condition.condition_id,), predecessor_need_ids=('n20',) if index == 60 else ())
        history.append(TurnRecord(index, question, 'Actual answer.'))
        states.append(TaskState(f'task-{index}', question, conditions=(condition,), needs=(need,),
                                current_turn_id=index))
    update = 'Use the approved group instead.'
    replacement = Condition('c61', value_spans=(TextRef(61, 'user', 8, 16),),
                            target_refs=('c60',), supersedes=('c60',), update='REPLACE')
    history.append(TurnRecord(61, update, 'Actual correction.'))
    from dataclasses import replace
    states.append(replace(states[-1], raw_question=update, conditions=(replacement,), current_turn_id=61,
        needs=(replace(states[-1].needs[0], condition_ids=('c61',)),)))
    memory = RetrievalState(61, tuple(history), tuple(states))
    windows = tuple(HistoryWindow(TextRef(index, 'user', 0, len(history[index-1].user_text)),
        history[index-1].user_text, len(history[index-1].user_text), .5)
        for index in (*range(1, 16), 61))
    turn = SimpleNamespace(turn=62, task_id='current', query='Compare with the earlier group.')
    observed = {}

    def respond(request):
        request_body = json.loads(request.content)
        body = json.loads(request_body['messages'][-1]['content'])
        observed.update(body=body, messages=request_body['messages'], schema=request_body['response_format'])
        active = body['prior_states'][0]
        predecessor = body['prior_predecessor_needs'][0]
        value = {'relation': 'followup', 'resume_task_id': active['task_id'], 'history_spans': [],
            'conditions': {}, 'condition_removals': [],
            'version_intent': {'mode': 'current', 'temporal_relation': 'at', 'document_families': [],
                'version_ids': [], 'resolved_version_ids': [], 'time_refs': [], 'source_turns': []},
            'needs': {'n0': {'anchor_spans': [{'window_id': 'w0', 'boundary_tokens': [0, 0]}],
                'subject_refs': [], 'operation': 'compare', 'condition_ids': [active['conditions'][0]['condition_id']],
                'predecessor_need_ids': [predecessor['need_id']], 'scope': 'selected_sources'}}}
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(value)},
            'finish_reason': 'stop'}], 'usage': {'completion_tokens': 100}})

    client = OpenAICompletionClient(SelectorConfig('http://task-fixture', 'fixture', 'fixture',
        max_input_tokens=14000, max_new_tokens=512), tokenizer=Characters(), transport=httpx.MockTransport(respond))
    teacher = TeacherTaskModel(client)
    try:
        prediction = teacher.predict(turn, windows, memory)
        assert client.calls == 1
        assert client.count_input_tokens(observed['messages']) <= client.config.max_input_tokens
    finally:
        client.close()
    assert prediction.resume_task_id == 'task-60'
    assert prediction.needs[0].condition_ids == ('c61',)
    assert prediction.needs[0].predecessor_need_ids == ('n20',)
    body = observed['body']
    assert 0 < len(body['prior_states']) < len(states)
    condition_values = [value for condition in body['prior_condition_text'].values() for value in condition['values']]
    assert all(value in condition_values for value in ('approved', 'group-60', 'group-20'))
    visible_turns = {window['turn_id'] for window in body['visible_windows']}
    assert {20, 60, 61, 62} <= visible_turns
    version_turns = observed['schema']['json_schema']['schema']['$defs']['VersionIntent']['properties']['source_turns']['items']['enum']
    assert set(version_turns) <= visible_turns
    assert len(memory.prior_task_states) == 61
    from jsonschema import Draft202012Validator
    schema = observed['schema']['json_schema']['schema']
    validator = Draft202012Validator({'$defs': schema['$defs'], '$ref': '#/$defs/Condition_0'})
    anchor = {'window_id': 'w0', 'boundary_tokens': [0, 0]}
    condition = {'key_span': None, 'value_spans': [], 'relation_type': 'PREDICATE',
        'predicate_span': None, 'target_refs': [], 'supersedes': [], 'update': 'ADD'}
    assert not validator.is_valid(condition)
    for field in ('key_span', 'predicate_span', 'value_spans'):
        complete = {**condition, field: [anchor] if field == 'value_spans' else anchor}
        validator.validate(complete)
        for required in condition:
            assert not validator.is_valid({key: value for key, value in complete.items() if key != required})
    for variant in schema['$defs']['Condition_0']['anyOf']:
        assert variant['type'] == 'object'
        assert set(variant['required']) == set(condition)
        assert not variant['additionalProperties']
    assert not validator.is_valid({'key_span': anchor})
    assert not validator.is_valid({**condition, 'key_span': anchor, 'target_refs': ['C0']})
    inherited = Draft202012Validator({'$defs': schema['$defs'], '$ref': '#/$defs/Condition_1'})
    target = body['prior_states'][0]['conditions'][0]['condition_id']
    inherited.validate({**condition, 'predicate_span': anchor, 'target_refs': [target], 'update': 'REPLACE'})
    assert not inherited.is_valid({**condition, 'key_span': anchor, 'target_refs': ['unrelated']})
