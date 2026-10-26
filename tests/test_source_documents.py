from dataclasses import replace
import json

from adapters.corpus import CorpusIndex
from adapters.mtrag import OnlineContext, OnlineTurn
from adapters.source_documents import SourceDocuments
from bench.quality import official_record
from controller.candidates import build_candidates
from controller.schema import SessionSnapshot
from controller.templates import verify_evidence_spans


def test_original_paragraph_window_preserves_formatting_and_shared_candidate_sources():
    text = ('Document introduction.\n\nProcedure A\n\nFor platform X:\n\n  run-x\n\n'
            'Only for this procedure, enable the flag.\n\nProcedure B\n\n'
            '1.  Download the package.\n2.  Run its installer.\n\nEnd of document.')
    passage = 'Only for this procedure, enable the flag.\n\nProcedure B\n\n1. Download the package.'
    documents = SourceDocuments([{'document_id':'manual','text':text,'url':'https://source/manual'}])
    corpus = CorpusIndex.from_records([{'passage_id':'part','document_id':'manual','text':passage,
        'span':[0,len(passage)],'url':'https://source/manual'}],corpus_id='c',snapshot_id='v1')
    context = OnlineContext('part',passage,score=.8)
    windows=corpus.document_windows((context,),documents)
    assert len(windows)==1 and context.text==passage
    support=corpus.enrich(windows)[0]
    assert support.text==text[support.span[0]:support.span[1]]
    assert 'For platform X:' in support.text and '2.  Run its installer.' in support.text
    assert corpus.document_windows((context,),documents)==windows
    assert corpus.document_window_counts=={'aligned':1}
    evidence=corpus.enrich((context,*windows))
    candidate,=build_candidates(SessionSnapshot('s'),'Which procedure applies?',evidence)
    assert verify_evidence_spans(candidate.messages,candidate.evidence_spans,evidence,candidate.reference_map)=={b.key for b in evidence}
    assert candidate.required_blocks==candidate.visible_blocks


def test_alignment_does_not_guess_document_or_choose_an_ambiguous_occurrence():
    documents=SourceDocuments([{'document_id':'doc','text':'Repeated.\n\nOther text.\n\nRepeated.'}])
    assert documents.window('absent','Repeated.')[1]=='document_unavailable'
    assert documents.window('doc','Changed original.')[1]=='passage_not_aligned'
    assert documents.window('doc','Repeated.')[1]=='ambiguous_alignment'
    assert documents.window('doc','Repeated. Other text. Repeated.')[1]=='already_complete'


def test_expanded_generation_export_preserves_distinct_raw_retrieval():
    original=OnlineContext('original','Ranked original',score=5)
    support=OnlineContext('source-window:doc:0:99','Complete original source window')
    turn=OnlineTurn('s','q',1,'cloud','Question',(),(original,),'P3')
    plain=official_record(turn,'Answer')
    expanded=official_record(replace(turn,supporting_contexts=(support,)),'Answer')
    assert 'retrieval_contexts' not in plain
    assert expanded['retrieval_contexts']==plain['contexts']
    assert [c['text'] for c in expanded['contexts']]==[original.text,support.text]
    assert expanded['contexts'][1]['score']==0


def test_source_documents_zip_and_plain_jsonl_load_the_same_originals(tmp_path):
    import zipfile
    path=tmp_path/'documents.jsonl'
    row={'document_id':'doc','text':'Heading\n\nA condition.\n\nActual statement.'}
    path.write_text(json.dumps(row)+'\n')
    archive=tmp_path/'documents.zip'
    with zipfile.ZipFile(archive,'w') as output:
        output.write(path,arcname=path.name)
    assert SourceDocuments.load(path).documents==SourceDocuments.load(archive).documents


def test_overlapping_original_windows_merge_without_reordering_ranked_passages():
    text='\n\n'.join(f'Paragraph {i} contains original conditions.' for i in range(9))
    documents=SourceDocuments([{'document_id':'manual','text':text}])
    rows=[{'passage_id':f'p{i}','document_id':'manual','span':[i,i+1],
           'text':f'Paragraph {i} contains original conditions.'} for i in (3,4)]
    corpus=CorpusIndex.from_records(rows,corpus_id='c',snapshot_id='v1')
    contexts=tuple(OnlineContext(r['passage_id'],r['text'],score=8-i) for i,r in enumerate(rows))
    support,=corpus.document_windows(contexts,documents)
    assert support.text==text[text.index('Paragraph 1'):text.index('\n\nParagraph 7')]
    assert [c.document_id for c in contexts]==['p3','p4']
    blocks=corpus.enrich((*contexts,support))
    from test_candidates import committed
    before=SessionSnapshot('s')
    first,=build_candidates(before,'Initial',blocks)
    snapshot=committed(first,before,'Initial',blocks)
    for candidate in build_candidates(snapshot,'Follow-up',blocks):
        assert candidate.required_blocks=={b.key for b in blocks}
        assert verify_evidence_spans(candidate.messages,candidate.evidence_spans,blocks,candidate.reference_map)==candidate.visible_blocks


def test_reading_view_replaces_unscoped_slices_but_keeps_declared_dependencies():
    documents=SourceDocuments([{'document_id':'manual','text':'Context.\n\nPlain fact.\n\nScoped fact.\n\nEnd.'}])
    corpus=CorpusIndex.from_records([
        {'passage_id':'plain','document_id':'manual','text':'Plain fact.','span':[0,1]},
        {'passage_id':'scoped','document_id':'manual','text':'Scoped fact.','span':[2,3],'dependencies':['condition']},
        {'passage_id':'condition','document_id':'conditions','text':'Only under the declared condition.'}],
        corpus_id='c',snapshot_id='v1')
    raw=(OnlineContext('plain','Plain fact.'),OnlineContext('scoped','Scoped fact.'))
    reading=corpus.reading_contexts(raw,documents)
    assert len(reading)==2 and reading[0].document_id.startswith('source-window:')
    assert reading[1] is raw[1]
    blocks=corpus.enrich(reading)
    assert {b.key.passage_id for b in blocks}=={reading[0].document_id,'scoped','condition'}
    assert corpus.reading_contexts((raw[0],raw[0]),documents)==reading[:1]
    turn=OnlineTurn('s','q',1,'c','Question',(),reading,'P3',retrieval_contexts=raw)
    exported=official_record(turn,'Actual answer')
    assert [c['document_id'] for c in exported['retrieval_contexts']]==['plain','scoped']
    assert [c['document_id'] for c in exported['contexts']]==[c.document_id for c in reading]


def test_shifted_retrieval_inside_one_paragraph_reuses_the_same_reading_prefix():
    from test_candidates import committed
    text='Introduction.\n\nFirst original sentence. Second original sentence.\n\nClosing context.'
    documents=SourceDocuments([{'document_id':'doc','text':text}])
    corpus=CorpusIndex.from_records([
        {'passage_id':'a','document_id':'doc','text':'First original sentence.','span':[0,1]},
        {'passage_id':'b','document_id':'doc','text':'Second original sentence.','span':[2,3]}],
        corpus_id='c',snapshot_id='v1')
    first=corpus.reading_contexts((OnlineContext('a','First original sentence.'),),documents)
    second=corpus.reading_contexts((OnlineContext('b','Second original sentence.'),),documents)
    assert first==second
    blocks=corpus.enrich(first)
    snapshot=SessionSnapshot('s')
    initial,=build_candidates(snapshot,'First question',blocks)
    snapshot=committed(initial,snapshot,'First question',blocks)
    append,rebuild=build_candidates(snapshot,'Follow-up',corpus.enrich(second))
    assert append.required_blocks==rebuild.required_blocks
    assert append.messages[:-1]==snapshot.execution.messages
    assert text not in append.messages[-1].content and text in rebuild.messages[-1].content


def test_reading_window_retains_parameter_anchors_for_later_revisits():
    documents=SourceDocuments([{'document_id':'doc','text':'Settings.\n\nSet `request_timeout` to 30 seconds.\n\nOther context.'}])
    corpus=CorpusIndex.from_records([{'passage_id':'p','document_id':'doc','span':[0,1],
        'text':'Set `request_timeout` to 30 seconds.'}],corpus_id='c',snapshot_id='v1')
    reading=corpus.reading_contexts((OnlineContext('p','Set `request_timeout` to 30 seconds.'),),documents)
    block,=corpus.enrich(reading)
    assert block.anchors==('request_timeout',)
    assert block.key.passage_id.startswith('source-window:v1:')
