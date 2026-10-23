from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

from bench.lettuce_judge import LettuceScorer, evaluate_audit, evaluate_file, format_prompt, main
from controller.templates import token_hash


def audit(answer="The limit is 8 MiB [E_01]."):
    return {"task_id":"t1", "conversation_id":"s1", "query":"Limit?", "answer":answer,
        "actual_history":[{"speaker":"assistant","text":"HISTORY_IS_NOT_SOURCE 99 MiB"}],
        "retrieved_evidence":[],
        "audit_evidence":[{"key":{"document_id":"a"},"text":"The limit is 8 MiB."},
                          {"key":{"document_id":"b"},"text":"Other full visible evidence."}],
        "visible_evidence_manifest":[{"reference_id":"E_01","key":{"document_id":"a"}},
                                     {"reference_id":"E_02","key":{"document_id":"b"}}],
        "input_ids":[1,2,3], "measurement_kind":"simulation", "truncated":False,
        "targets":[{"text":"GOLD_SECRET"}], "offline_expected":{"label":"LABEL_SECRET"},
        "source_arm_for_offline_review":"ARM_SECRET"}


class PairEncoding(dict):
    def __init__(self, values, sequence_ids):
        super().__init__(values)
        self._sequences=sequence_ids

    def sequence_ids(self):
        return self._sequences


class PairTokenizer:
    def __init__(self):
        self.calls=[]

    def __call__(self, prompt, answer, **kwargs):
        self.calls.append((prompt,answer,kwargs))
        offsets=[(m.start(),m.end()) for m in re.finditer(r"\S+",answer)]
        return PairEncoding({"input_ids":[101,200,102]+list(range(10,10+len(offsets)))+[102],
            "attention_mask":[1]*(len(offsets)+4),
            "offset_mapping":[(0,0),(0,len(prompt)),(0,0)]+offsets+[(0,0)],
            "special_tokens_mask":[1,0,1]+[0]*len(offsets)+[1]},
            [None,0,None]+[1]*len(offsets)+[None])


class Predictions:
    def __init__(self, unsupported=()):
        self.unsupported=set(unsupported)
        self.calls=[]

    def __call__(self, encoded):
        self.calls.append(encoded)
        return [[.1,.9] if i in self.unsupported else [.9,.1] for i in encoded["input_ids"]]


def test_author_qa_template_and_pair_format_are_fixed():
    expected=('Briefly answer the following question:\nQ?\n'
              'Bear in mind that your response should be strictly based on the following 2 passages:\n'
              'passage 1: A\npassage 2: B\n'
              'In case the passages do not contain the necessary information to answer the question, '
              'please reply with: "Unable to answer based on given passages."\noutput:')
    assert format_prompt(["A","B"],"Q?")==expected


def test_full_visible_sources_and_answer_preserved_without_label_or_history_leak():
    row=audit()
    before=deepcopy(row)
    tokenizer, predictions=PairTokenizer(),Predictions()
    out=evaluate_audit(row,LettuceScorer(tokenizer,predictions))
    assert row==before and out["status"]=="scored"
    assert len(predictions.calls)==1 and out["chunk_count"]==1
    prompt,answer,kwargs=tokenizer.calls[0]
    assert answer==row["answer"] and "[E_01]" in answer
    assert all(b["text"] in prompt for b in row["audit_evidence"])
    assert all(x not in prompt+answer for x in ["GOLD_SECRET","LABEL_SECRET","ARM_SECRET","HISTORY_IS_NOT_SOURCE"])
    assert kwargs["truncation"] is False and kwargs["padding"] is False
    assert out["input_hash"]==token_hash(row["input_ids"])
    assert out["history_turns_not_submitted"]==1
    assert out["formal_quality_eligible"] is False and "scores" not in out and "verdict" not in out
    assert out["semantic_quality_score"] is None and out["citation_correctness_score"] is None


def test_visible_order_and_duplicate_material_are_not_silently_changed():
    row=audit()
    row["visible_evidence_manifest"].append({"reference_id":"E_03","key":{"document_id":"a"}})
    out=evaluate_audit(row)
    assert out["model_input"]["prompt"].count("The limit is 8 MiB.")==2
    assert [s["reference_id"] for s in out["visible_sources"]]==["E_01","E_02","E_03"]


def test_sequence_ids_unicode_leading_whitespace_and_special_tokens_keep_offsets():
    answer="  élan café 漢字"
    tokenizer, predictions=PairTokenizer(),Predictions([101,200,102,10,11])
    out=evaluate_audit(audit(answer),LettuceScorer(tokenizer,predictions))
    assert [t["text"] for t in out["tokens"]]==["élan","café","漢字"]
    assert [t["start"] for t in out["tokens"]]==[2,7,12]
    assert out["unsupported_spans"]==[{"start":2,"end":11,"confidence":.9,"token_count":2,
                                        "text":"élan café","citation_marker_intersections":[]}]
    assert all(answer[t["start"]:t["end"]]==t["text"] for t in out["tokens"])
    assert out["coverage"]["unscored_nonwhitespace_character_positions"]==[]


def test_citation_related_predictions_are_marked_not_deleted_or_scored_as_correctness():
    row=audit("Fact [E_01].")
    out=evaluate_audit(row,LettuceScorer(PairTokenizer(),Predictions([11])))
    token=out["tokens"][-1]
    assert token["overlaps_citation_marker"] is True
    span=out["unsupported_spans"][0]
    assert span["text"]=="[E_01]." and span["citation_marker_intersections"]==[{"start":5,"end":11}]
    assert out["citation_correctness_score"] is None


def test_argmax_tie_uses_class_zero_and_no_span_is_not_quality_pass():
    out=evaluate_audit(audit("Fact"),LettuceScorer(PairTokenizer(),lambda e:[[.5,.5]]*len(e["input_ids"])))
    assert out["tokens"][0]["label_id"]==0 and out["unsupported_spans"]==[]
    assert out["unsupported_detected"] is False and out["semantic_quality_score"] is None


def test_over_limit_counts_complete_pair_without_inference_or_truncation():
    tokenizer,predictions=PairTokenizer(),Predictions()
    row=audit("one two three")  # 7 tokens including native special/context tokens
    out=evaluate_audit(row,LettuceScorer(tokenizer,predictions,max_length=6))
    assert out["input_tokens"]==7 and out["status"]=="not_covered"
    assert out["reason"]=="complete_input_exceeds_context" and not predictions.calls
    assert out["model_inference_attempted"] is False and out["truncated"] is False
    assert out["model_input"]["answer"]==row["answer"]
    at_limit=evaluate_audit(row,LettuceScorer(PairTokenizer(),Predictions(),max_length=7))
    assert at_limit["status"]=="scored"


@pytest.mark.parametrize("change,reason", [
    ({"answer":" \n"},"empty_answer"),
    ({"visible_evidence_manifest":[]},"no_visible_sources"),
    ({"audit_evidence":[]},"missing_or_empty_visible_evidence_body"),
    ({"audit_evidence":[{"key":{"document_id":"a"},"text":""}]},"missing_or_empty_visible_evidence_body"),
])
def test_empty_and_missing_material_remain_uncovered_without_model(change,reason):
    predictions=Predictions()
    out=evaluate_audit(audit()|change,LettuceScorer(PairTokenizer(),predictions))
    assert out["status"]=="not_covered" and out["reason"]==reason and not predictions.calls


def test_failed_forward_keeps_row_and_later_row_is_still_exported(tmp_path):
    calls=0
    def forward(encoded):
        nonlocal calls
        calls+=1
        if calls==1:
            raise RuntimeError("synthetic forward failure")
        return [[.9,.1]]*len(encoded["input_ids"])
    source,output=tmp_path/"audit.jsonl",tmp_path/"ledger.jsonl"
    source.write_text(json.dumps(audit())+"\n"+json.dumps(audit()|{"task_id":"t2"})+"\n")
    evaluate_file(source,output,LettuceScorer(PairTokenizer(),forward))
    rows=[json.loads(s) for s in output.read_text().splitlines()]
    assert [r["status"] for r in rows]==["failed","scored"]
    assert rows[0]["model_inference_attempted"] is True and rows[0]["model_inference_performed"] is False
    assert "synthetic forward failure" in rows[0]["error"] and rows[0]["input_tokens"]>0
    with pytest.raises(FileExistsError):
        evaluate_file(source,output)


def test_prediction_alignment_error_does_not_silently_drop_answer_tokens():
    out=evaluate_audit(audit(),LettuceScorer(PairTokenizer(),lambda e:[]))
    assert out["status"]=="failed" and "align" in out["error"]
    assert out["model_inference_performed"] is True


def test_out_of_answer_offsets_cannot_fabricate_span_text():
    class BadOffsets(PairTokenizer):
        def __call__(self,*args,**kwargs):
            value=super().__call__(*args,**kwargs)
            value["offset_mapping"][3]=(0,10000)
            return value
    out=evaluate_audit(audit(),LettuceScorer(BadOffsets(),Predictions([10])))
    assert out["status"]=="failed" and "outside" in out["error"]


def test_nonwhitespace_without_a_token_is_partial_coverage():
    class Gap(PairTokenizer):
        def __call__(self,*args,**kwargs):
            value=super().__call__(*args,**kwargs)
            value["special_tokens_mask"][3]=1
            return value
    out=evaluate_audit(audit("First second"),LettuceScorer(Gap(),Predictions()))
    assert out["status"]=="partial" and out["coverage"]["unscored_nonwhitespace_character_positions"]==list(range(5))


def test_cli_without_local_model_only_exports_and_does_not_load_weights(tmp_path,monkeypatch):
    source,output=tmp_path/"audit.jsonl",tmp_path/"out.jsonl"
    source.write_text(json.dumps(audit())+"\n")
    def forbidden(*args,**kwargs):
        raise AssertionError("model-free export loaded a model")
    monkeypatch.setattr(LettuceScorer,"from_local_snapshot",forbidden)
    main(["--audit",str(source),"--output",str(output)])
    out=json.loads(output.read_text())
    assert out["status"]=="dry_run" and out["model_inference_performed"] is False


def test_assets_preserve_upstream_license_and_model_identity():
    root=Path(__file__).resolve().parents[1]/"bench/assets/lettuce_judge"
    p=json.loads((root/"protocol.json").read_text())
    assert p["model_revision"]=="8831ce063dff760b01efe4519b4188f5ee2456f2"
    assert p["label_direction"]=={"0":"supported","1":"hallucinated"}
    assert "Copyright (c) 2025 KR Labs" in (root/"LICENSE").read_text()
    assert "MIT" in (root/"NOTICE.md").read_text()


def test_local_loader_never_downloads_and_checks_native_label_direction(tmp_path,monkeypatch):
    loads=[]
    model=SimpleNamespace(config=SimpleNamespace(id2label={0:"supported",1:"hallucinated"},
        max_position_embeddings=8192,architectures=["ModernBertForTokenClassification"]),
        device="cpu",dtype="float32")
    model.to=lambda device:model
    model.eval=lambda:model
    class TokenizerLoader:
        @staticmethod
        def from_pretrained(path,**kwargs):
            loads.append(("tokenizer",path,kwargs))
            return SimpleNamespace(model_max_length=8192)
    class ModelLoader:
        @staticmethod
        def from_pretrained(path,**kwargs):
            loads.append(("model",path,kwargs))
            return model
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace())
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(
        AutoTokenizer=TokenizerLoader,AutoModelForTokenClassification=ModelLoader))
    scorer=LettuceScorer.from_local_snapshot(tmp_path)
    assert scorer.max_length==8192 and scorer.provenance["model_downloads"] is False
    assert all(args=={"local_files_only":True,"trust_remote_code":False} for _,_,args in loads)
    model.config.id2label={0:"hallucinated",1:"supported"}
    with pytest.raises(ValueError,match="label direction"):
        LettuceScorer.from_local_snapshot(tmp_path)


def test_real_fast_tokenizer_pair_offsets_without_model_or_download():
    tokenizers=pytest.importorskip("tokenizers")
    transformers=pytest.importorskip("transformers")
    backend=tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[UNK]":0,"[CLS]":1,"[SEP]":2,"élan":3,"café":4,"漢字":5},unk_token="[UNK]"))
    backend.pre_tokenizer=tokenizers.pre_tokenizers.WhitespaceSplit()
    backend.post_processor=tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]",pair="[CLS] $A [SEP] $B:1 [SEP]:1",special_tokens=[("[CLS]",1),("[SEP]",2)])
    tokenizer=transformers.PreTrainedTokenizerFast(tokenizer_object=backend,
        unk_token="[UNK]",cls_token="[CLS]",sep_token="[SEP]",model_max_length=8192)
    out=evaluate_audit(audit("  élan café 漢字"),LettuceScorer(tokenizer,Predictions([3,4])))
    assert out["status"]=="scored"
    assert out["unsupported_spans"][0]["text"]=="élan café"
    assert out["unsupported_spans"][0]["start"]==2
